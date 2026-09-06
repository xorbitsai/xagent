"""Tests for web tracer factory helpers."""

from __future__ import annotations

from typing import cast

import pytest

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.agent.trace import TraceEvent, TraceHandler
from xagent.core.tracing.langfuse.client import get_langfuse_client
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler
from xagent.web.models.user import User
from xagent.web.tracing import create_ephemeral_tracer, create_task_tracer


@pytest.fixture(autouse=True)
def legacy_task_selection(monkeypatch):
    from xagent.web.api.trace_handlers import DatabaseTraceHandler

    monkeypatch.setattr(
        "xagent.web.tracing.task_database_handler", DatabaseTraceHandler
    )


class DummyTraceHandler(TraceHandler):
    async def handle_event(self, event: TraceEvent) -> None:
        del event


def test_create_task_tracer_without_langfuse(langfuse_client_reset):
    tracer = create_task_tracer(123)
    handler_names = [type(handler).__name__ for handler in tracer.handlers]
    assert handler_names == [
        "ConsoleTraceHandler",
        "DatabaseTraceHandler",
        "WebSocketTraceHandler",
    ]


def test_create_task_tracer_with_langfuse(mocker, monkeypatch, langfuse_client_reset):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    create_langfuse_mock(mocker)

    user = User()
    user.id = 99

    tracer = create_task_tracer(123, user)
    assert any(isinstance(handler, LangfuseTraceHandler) for handler in tracer.handlers)


def test_create_task_tracer_with_user_id(mocker, monkeypatch, langfuse_client_reset):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    create_langfuse_mock(mocker)

    tracer = create_task_tracer(123, user_id=42)
    langfuse_handler = next(
        handler
        for handler in tracer.handlers
        if isinstance(handler, LangfuseTraceHandler)
    )

    assert langfuse_handler.user_id == "42"


def test_create_ephemeral_tracer_with_langfuse(
    mocker, monkeypatch, langfuse_client_reset
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    create_langfuse_mock(mocker)

    handler = DummyTraceHandler()
    tracer = create_ephemeral_tracer(
        task_id="preview-task",
        websocket_handler=handler,
        is_preview=True,
        user=cast(User, None),
    )

    assert tracer.handlers[0] is handler
    assert any(
        isinstance(candidate, LangfuseTraceHandler) for candidate in tracer.handlers
    )


def test_langfuse_client_prefers_base_url(mocker, monkeypatch, langfuse_client_reset):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://base-url.example")
    monkeypatch.setenv("LANGFUSE_HOST", "https://legacy-host.example")
    mock_langfuse_class, mock_langfuse = create_langfuse_mock(mocker)

    client = get_langfuse_client()

    assert client is mock_langfuse
    mock_langfuse_class.assert_called_once_with(base_url="https://base-url.example")


def test_langfuse_client_uses_host_fallback(mocker, monkeypatch, langfuse_client_reset):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://legacy-host.example")
    mock_langfuse_class, mock_langfuse = create_langfuse_mock(mocker)

    client = get_langfuse_client()

    assert client is mock_langfuse
    mock_langfuse_class.assert_called_once_with(base_url="https://legacy-host.example")
