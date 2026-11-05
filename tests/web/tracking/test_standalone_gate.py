"""``usage_scope`` checks the standalone gate before its work reaches a provider."""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from xagent.core.model.chat.token_context import (
    MediaCallType,
    add_media_usage,
    token_context,
)
from xagent.web.services import quota_hooks
from xagent.web.tracking.standalone_usage import (
    StandaloneUsageRefused,
    check_standalone_gate,
    run_in_usage_scope,
    usage_scope,
)


class _Gate:
    def __init__(self, reason: Any = None) -> None:
        self.reason = reason
        self.calls: list[tuple[int, str]] = []
        self.threads: list[int] = []

    def __call__(self, user_id: int, source: str) -> Any:
        self.calls.append((user_id, source))
        self.threads.append(threading.get_ident())
        if isinstance(self.reason, BaseException):
            raise self.reason
        return self.reason


@pytest.fixture
def usage():
    calls: list = []
    quota_hooks.set_standalone_usage_hook(lambda *args: calls.append(args))
    yield calls
    quota_hooks.set_standalone_usage_hook(None)


@pytest.fixture
def gate():
    recorder = _Gate()
    quota_hooks.set_standalone_gate_hook(recorder)
    yield recorder
    quota_hooks.set_standalone_gate_hook(None)


def _embed() -> None:
    add_media_usage(call_type=MediaCallType.EMBEDDING, quantity=1, model_id="e")


def test_an_allowing_gate_is_asked_once_and_the_work_runs(gate, usage):
    with usage_scope(7, usage_source="kb_search"):
        _embed()

    assert gate.calls == [(7, "kb_search")]
    assert [(user_id, source) for user_id, _rows, source in usage] == [(7, "kb_search")]


def test_a_refusal_stops_the_work_before_it_is_bound_or_metered(gate, usage):
    gate.reason = "Out of AI credits"
    ran = []
    before = token_context.get(None)

    with pytest.raises(StandaloneUsageRefused) as refused:
        with usage_scope(7, usage_source="memory_api"):
            ran.append(True)

    assert ran == [] and usage == []
    assert token_context.get(None) is before
    exc = refused.value
    assert exc.status_code == 402
    assert exc.detail == {
        "code": "standalone_usage_refused",
        "source": "memory_api",
        "message": "Out of AI credits",
        "reason": "Out of AI credits",
    }
    assert str(exc) == "Out of AI credits"


def test_a_structured_reason_is_forwarded_as_is(gate):
    gate.reason = {"code": "ai_credits_exhausted", "message": "Top up", "plan": "free"}

    with pytest.raises(StandaloneUsageRefused) as refused:
        check_standalone_gate(7, "speech_synthesize", ())

    detail = refused.value.detail
    assert detail["code"] == "standalone_usage_refused"
    assert detail["reason"] == gate.reason
    assert detail["message"] == "Top up" == str(refused.value)


def test_a_structured_reason_without_a_message_gets_a_generic_one(gate):
    gate.reason = {"code": "ai_credits_exhausted"}

    with pytest.raises(StandaloneUsageRefused) as refused:
        check_standalone_gate(7, "kb_search", ())

    assert refused.value.message == "Usage limit reached for this request"


async def test_async_scope_asks_the_gate_off_the_event_loop(gate):
    gate.reason = "no"
    ran = []

    with pytest.raises(StandaloneUsageRefused):
        async with usage_scope(3, usage_source="speech_transcribe"):
            ran.append(True)

    assert ran == []
    assert gate.calls == [(3, "speech_transcribe")]
    assert gate.threads[0] != threading.get_ident()


def test_run_in_usage_scope_refuses_before_calling(gate):
    gate.reason = "no"
    called = []

    with pytest.raises(StandaloneUsageRefused):
        run_in_usage_scope(5, "kb_ingest", (), called.append, 1)

    assert called == []


@pytest.mark.parametrize(
    "user_id,source", [(None, "telegram_voice"), (7, "model_probe")]
)
def test_unattributed_work_and_model_probes_skip_the_gate(gate, usage, user_id, source):
    gate.reason = "no"

    with usage_scope(user_id, usage_source=source):
        _embed()

    assert gate.calls == []
    assert len(usage) == 1


def test_a_failing_gate_allows_the_work_and_logs(gate, caplog):
    gate.reason = RuntimeError("plan lookup failed")
    ran = []

    with caplog.at_level(logging.WARNING):
        with usage_scope(7, usage_source="kb_search"):
            ran.append(True)

    assert ran == [True]
    assert "Standalone gate failed for kb_search" in caplog.text


async def test_without_a_gate_nothing_is_checked(usage):
    async with usage_scope(7, usage_source="kb_search"):
        _embed()

    check_standalone_gate(7, "kb_search", ("platform/text-embedding-v4",))
    assert len(usage) == 1


class _ModelGate:
    def __init__(self, reason: Any = None) -> None:
        self.reason = reason
        self.calls: list[tuple[int, str, Any]] = []

    def __call__(self, user_id: int, source: str, model_ids: Any) -> Any:
        self.calls.append((user_id, source, model_ids))
        return self.reason


@pytest.fixture
def model_gate():
    recorder = _ModelGate()
    quota_hooks.set_standalone_gate_hook(recorder)
    yield recorder
    quota_hooks.set_standalone_gate_hook(None)


async def test_a_three_argument_gate_gets_the_model_ids(model_gate):
    # Empty ids (an environment-configured model) and repeats are dropped.
    async with usage_scope(
        7,
        usage_source="kb_search",
        model_ids=("platform/text-embedding-v4", "", "platform/text-embedding-v4"),
    ):
        pass
    with usage_scope(7, usage_source="kb_search", model_ids=("", "byo-rerank")):
        pass
    run_in_usage_scope(7, "kb_ingest", ("byo-embed",), lambda: None)

    assert model_gate.calls == [
        (7, "kb_search", ("platform/text-embedding-v4",)),
        (7, "kb_search", ("byo-rerank",)),
        (7, "kb_ingest", ("byo-embed",)),
    ]


def test_the_gate_refuses_by_model_id(model_gate):
    quota_hooks.set_standalone_gate_hook(
        lambda user_id, source, model_ids: (
            "Out of AI credits"
            if any(m.startswith("platform/") for m in model_ids)
            else None
        )
    )
    with usage_scope(7, usage_source="memory_api", model_ids=("byo-embed",)):
        pass
    with pytest.raises(StandaloneUsageRefused):
        with usage_scope(7, usage_source="memory_api", model_ids=("platform/x",)):
            pass


def test_a_two_argument_gate_keeps_its_signature(gate):
    # Registered with the pre-model_ids signature (the fixture's _Gate).
    with usage_scope(7, usage_source="kb_search", model_ids=("platform/x",)):
        pass
    check_standalone_gate(7, "memory_api", ("platform/x",))

    assert gate.calls == [(7, "kb_search"), (7, "memory_api")]
    assert quota_hooks.get_standalone_gate_hook() is gate


def test_a_gate_with_neither_signature_is_rejected_at_registration():
    with pytest.raises(TypeError):
        quota_hooks.set_standalone_gate_hook(lambda user_id: None)
    assert quota_hooks.get_standalone_gate() is None
