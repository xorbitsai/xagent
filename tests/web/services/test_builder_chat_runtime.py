from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.services import builder_chat_runtime
from xagent.web.services.builder_chat_runtime import (
    BuilderChatRuntimeInputs,
    load_builder_chat_runtime_inputs,
)


@pytest.mark.asyncio
async def test_load_builder_chat_runtime_inputs_uses_worker_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    session_events: list[tuple[str, int]] = []
    selected_llm = object()
    default_compact_llm = object()

    class FakeQuery:
        def filter(self, *_conditions: object) -> FakeQuery:
            return self

        def all(self) -> list[object]:
            session_events.append(("query", threading.get_ident()))
            return [SimpleNamespace(file_id="owned-file")]

    class FakeSession:
        def __enter__(self) -> FakeSession:
            session_events.append(("enter", threading.get_ident()))
            return self

        def __exit__(self, *_args: object) -> None:
            session_events.append(("exit", threading.get_ident()))

        def query(self, _model: object) -> FakeQuery:
            return FakeQuery()

    class FakeStorage:
        def __init__(self, _session: FakeSession) -> None:
            session_events.append(("storage", threading.get_ident()))

        def get_llm_by_name_with_access(
            self, model_name: object, user_id: int | None = None
        ) -> object | None:
            assert user_id == 42
            assert model_name == "selected"
            return selected_llm

        def get_configured_defaults(
            self,
            user_id: int | None = None,
            *,
            config_types: tuple[str, ...],
            fallback_llm: object,
        ) -> tuple[None, None, None, object]:
            assert user_id == 42
            assert config_types == ("compact",)
            assert fallback_llm is selected_llm
            return None, None, None, default_compact_llm

    monkeypatch.setattr(
        builder_chat_runtime,
        "get_session_local",
        lambda: FakeSession,
    )
    monkeypatch.setattr(
        builder_chat_runtime,
        "UserAwareModelStorage",
        FakeStorage,
    )

    result = await load_builder_chat_runtime_inputs(
        user_id=42,
        requested_file_ids=("owned-file", "missing-file", "owned-file"),
        model_name="selected",
        compact_model_name=None,
    )

    assert result == BuilderChatRuntimeInputs(
        authorized_file_ids=("owned-file", "owned-file"),
        llm=selected_llm,
        compact_llm=default_compact_llm,
    )
    assert [event for event, _thread_id in session_events] == [
        "enter",
        "query",
        "storage",
        "exit",
    ]
    assert all(thread_id != event_loop_thread for _event, thread_id in session_events)


@pytest.mark.asyncio
async def test_load_builder_chat_runtime_inputs_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    worker_started = threading.Event()
    allow_worker = threading.Event()

    def blocking_load(**_kwargs: Any) -> BuilderChatRuntimeInputs:
        worker_started.set()
        assert threading.get_ident() != event_loop_thread
        assert allow_worker.wait(timeout=30)
        return BuilderChatRuntimeInputs(
            authorized_file_ids=(),
            llm=object(),
            compact_llm=None,
        )

    monkeypatch.setattr(
        builder_chat_runtime,
        "_load_builder_chat_runtime_inputs_sync",
        blocking_load,
    )

    load_task = asyncio.create_task(
        load_builder_chat_runtime_inputs(
            user_id=1,
            requested_file_ids=(),
            model_name=None,
            compact_model_name=None,
        )
    )
    try:
        assert await asyncio.to_thread(worker_started.wait, 30)
        await asyncio.sleep(0)
        assert not load_task.done()
    finally:
        allow_worker.set()
        await asyncio.wait_for(load_task, timeout=30)
