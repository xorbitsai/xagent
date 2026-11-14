"""Every built-in web search provider sends only the redacted query.

HTTP uses a real ``httpx`` client on ``httpx.MockTransport``; the Exa SDK call
is stubbed. All personal data below is invented.
"""

import json
import logging
import sys
import types
from typing import Any, Dict, List

import httpx
import pytest

from xagent.core.agent.trace import (
    ConsoleTraceHandler,
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.core.tools.core import (
    tavily_web_search,
    web_search,
    zhipu_web_search,
)
from xagent.core.tools.core.exa_web_search import ExaWebSearchCore
from xagent.core.tools.core.tavily_web_search import TavilyWebSearchCore
from xagent.core.tools.core.web_search import WebSearchCore
from xagent.core.tools.core.web_search_guard import (
    SearchAuditEvent,
    get_search_audit_hook,
    set_search_audit_hook,
)
from xagent.core.tools.core.zhipu_web_search import ZhipuWebSearchCore

EMAIL = "jane.doe@example.com"
PHONE = "+61 412 345 678"
ADDRESS = "12 Smith Street, Surry Hills NSW 2010"
TFN = "123 456 782"
RAW_VALUES = [EMAIL, PHONE, ADDRESS, TFN, "412 345 678", "Smith Street", "123456782"]
QUERY = f"find {EMAIL} {PHONE} {ADDRESS} TFN {TFN} Acme Pty Ltd"
REDACTED = "find [EMAIL] [PHONE] [ADDRESS] TFN [TFN] Acme Pty Ltd"
EXPECTED_ENTITIES = ("email", "address", "tfn", "phone")


@pytest.fixture(autouse=True)
def _clean_hook(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("ZHIPU_API_KEY", "k")
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    set_search_audit_hook(None)
    yield
    set_search_audit_hook(None)


class Wire:
    """Captures what the provider put on the wire."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []

    def text(self) -> str:
        return " ".join(f"{r.url} {r.content.decode()}" for r in self.requests)


def _install_http(monkeypatch, module, payload: Dict[str, Any], status: int = 200):
    wire = Wire()

    def handler(request: httpx.Request) -> httpx.Response:
        wire.requests.append(request)
        return httpx.Response(status, json=payload)

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("proxy", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    if module is zhipu_web_search:
        monkeypatch.setattr(
            module, "create_public_async_client", lambda **kw: factory()
        )
    else:
        monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    return wire


def _install_exa(monkeypatch):
    sent: List[Dict[str, Any]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            self.headers: Dict[str, str] = {}

        def search(self, **kwargs: Any):
            sent.append(kwargs)
            return types.SimpleNamespace(
                results=[types.SimpleNamespace(title="T", url="https://e.com")]
            )

    monkeypatch.setitem(sys.modules, "exa_py", types.SimpleNamespace(Exa=FakeExa))
    return sent


async def _run_provider(name: str, monkeypatch):
    """Run one provider with QUERY; return (text that reached the provider, results)."""
    if name == "google":
        wire = _install_http(
            monkeypatch,
            web_search,
            {"items": [{"title": "t", "link": "l", "snippet": "s"}]},
        )
        res = await WebSearchCore().search(QUERY)
        return wire.text(), res
    if name == "tavily":
        wire = _install_http(
            monkeypatch,
            tavily_web_search,
            {"results": [{"title": "t", "url": "u", "content": "c"}]},
        )
        res = await TavilyWebSearchCore().search(QUERY)
        return wire.text(), res
    if name == "zhipu":
        wire = _install_http(
            monkeypatch,
            zhipu_web_search,
            {"search_result": [{"title": "t", "link": "l", "content": "c"}]},
        )
        res = await ZhipuWebSearchCore().search(QUERY)
        return wire.text(), res
    sent = _install_exa(monkeypatch)
    res = await ExaWebSearchCore().search(QUERY)
    return json.dumps(sent), res


PROVIDERS = ["google", "tavily", "zhipu", "exa"]


def _decoded(text: str) -> str:
    from urllib.parse import unquote_plus

    return unquote_plus(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_provider_receives_placeholders_only(provider, monkeypatch):
    wire_text, results = await _run_provider(provider, monkeypatch)
    wire_text = _decoded(wire_text)

    assert len(results["search_result"] if isinstance(results, dict) else results) == 1
    assert REDACTED in wire_text
    for value in RAW_VALUES:
        assert value not in wire_text
    assert "Acme Pty Ltd" in wire_text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_info_logs_carry_no_raw_values(provider, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        await _run_provider(provider, monkeypatch)

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert logged  # the search did log something
    for value in RAW_VALUES:
        assert value not in logged
    assert "[EMAIL]" in logged  # redacted text is what gets logged


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_audit_hook_receives_redacted_query(provider, monkeypatch):
    events: List[SearchAuditEvent] = []
    set_search_audit_hook(events.append)

    await _run_provider(provider, monkeypatch)

    assert len(events) == 1
    event = events[0]
    assert event.provider == provider
    assert event.query == REDACTED
    assert set(event.entities) == set(EXPECTED_ENTITIES)
    assert event.result_count == 1
    assert event.outcome == "success"
    assert event.failure_kind is None
    for value in RAW_VALUES:
        assert value not in repr(event)


@pytest.mark.asyncio
async def test_async_hook_is_awaited(monkeypatch):
    events: List[SearchAuditEvent] = []

    async def hook(event: SearchAuditEvent) -> None:
        events.append(event)

    set_search_audit_hook(hook)
    await _run_provider("tavily", monkeypatch)
    assert len(events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_failing_hook_does_not_fail_search(provider, monkeypatch, caplog):
    def hook(event: SearchAuditEvent) -> None:
        raise RuntimeError(f"db down while writing {event.query} {EMAIL}")

    set_search_audit_hook(hook)
    with caplog.at_level(logging.INFO):
        _, results = await _run_provider(provider, monkeypatch)

    assert results
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "audit hook failed" in logged
    assert EMAIL not in logged
    assert "[EMAIL]" not in logged.split("audit hook failed")[1]


@pytest.mark.asyncio
async def test_hook_reports_failure_outcome(monkeypatch):
    events: List[SearchAuditEvent] = []
    set_search_audit_hook(events.append)
    _install_http(monkeypatch, tavily_web_search, {"detail": "boom"}, status=500)

    with pytest.raises(ValueError):
        await TavilyWebSearchCore().search(QUERY)

    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].failure_kind == "HTTPStatusError"
    assert events[0].result_count is None
    assert events[0].query == REDACTED


@pytest.mark.asyncio
async def test_no_hook_registered_by_default_and_search_unchanged(monkeypatch):
    assert get_search_audit_hook() is None
    _, results = await _run_provider("google", monkeypatch)
    assert results == [{"title": "t", "link": "l", "snippet": "s"}]


@pytest.mark.asyncio
async def test_query_without_personal_data_is_sent_unchanged(monkeypatch):
    wire = _install_http(monkeypatch, tavily_web_search, {"results": []})
    await TavilyWebSearchCore().search("Elon Musk Tesla 2024 price $1,234")
    assert json.loads(wire.requests[0].content)["query"] == (
        "Elon Musk Tesla 2024 price $1,234"
    )


# --- console trace -----------------------------------------------------------

_SHAPES = [
    (TraceScope.ACTION, TraceAction.START),
    (TraceScope.ACTION, TraceAction.END),
    (TraceScope.ACTION, TraceAction.ERROR),
    # every console sink renders ``event.data``, whatever the scope
    (TraceScope.TASK, TraceAction.UPDATE),
    (TraceScope.STEP, TraceAction.UPDATE),
    (TraceScope.SYSTEM, TraceAction.UPDATE),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name", ["web_search", "zhipu_web_search", "exa_web_search"]
)
@pytest.mark.parametrize("scope,action", _SHAPES)
async def test_console_trace_shows_redacted_query(tool_name, scope, action, caplog):
    event = TraceEvent(
        TraceEventType(scope, action, TraceCategory.TOOL),
        task_id="t1",
        step_id="s1",
        data={
            "tool_name": tool_name,
            "tool_params": {"query": QUERY, "num_results": 3},
            "tool_call_id": "c1",
        },
    )
    with caplog.at_level(logging.INFO):
        await ConsoleTraceHandler().handle_event(event)

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert REDACTED in logged
    for value in RAW_VALUES:
        assert value not in logged
    # the stored payload is untouched
    assert event.data["tool_params"]["query"] == QUERY


@pytest.mark.asyncio
async def test_console_trace_leaves_other_tools_alone(caplog):
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL),
        task_id="t1",
        step_id="s1",
        data={"tool_name": "write_file", "tool_params": {"query": EMAIL}},
    )
    with caplog.at_level(logging.INFO):
        await ConsoleTraceHandler().handle_event(event)
    assert EMAIL in "\n".join(r.getMessage() for r in caplog.records)


# --- redaction failure fails closed ------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "tavily", "zhipu"])
async def test_redactor_failure_sends_nothing(provider, monkeypatch, caplog):
    import xagent.core.tools.core.search_query_redaction as redaction
    from xagent.core.tools.core.web_search_guard import SearchQueryRedactionError

    events: List[SearchAuditEvent] = []
    set_search_audit_hook(events.append)
    monkeypatch.setattr(
        redaction, "_sub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(EMAIL))
    )
    sent: List[httpx.Request] = []
    module = {
        "google": web_search,
        "tavily": tavily_web_search,
        "zhipu": zhipu_web_search,
    }[provider]
    wire = _install_http(monkeypatch, module, {})
    core = {
        "google": WebSearchCore,
        "tavily": TavilyWebSearchCore,
        "zhipu": ZhipuWebSearchCore,
    }[provider]()

    with caplog.at_level(logging.INFO):
        with pytest.raises(SearchQueryRedactionError) as info:
            await core.search(QUERY)

    assert wire.requests == [] and sent == []
    assert EMAIL not in str(info.value) and "Acme" not in str(info.value)
    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].failure_kind == "redaction_error"
    assert events[0].query == "[REDACTED]"
    assert EMAIL not in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_redactor_failure_sends_nothing_exa(monkeypatch):
    import xagent.core.tools.core.search_query_redaction as redaction
    from xagent.core.tools.core.web_search_guard import SearchQueryRedactionError

    sent = _install_exa(monkeypatch)
    monkeypatch.setattr(
        redaction, "_sub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(SearchQueryRedactionError):
        await ExaWebSearchCore().search(QUERY)
    assert sent == []


# --- hook runs in the tool call's own context --------------------------------


@pytest.mark.asyncio
async def test_hook_sees_contextvars_of_the_search_call(monkeypatch):
    import contextvars

    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    seen: List[str] = []

    async def hook(event: SearchAuditEvent) -> None:
        seen.append(f"async:{marker.get()}")

    def sync_hook(event: SearchAuditEvent) -> None:
        seen.append(f"sync:{marker.get()}")

    token = marker.set("task-42")
    try:
        set_search_audit_hook(hook)
        with monkeypatch.context() as m:
            await _run_provider("tavily", m)
        set_search_audit_hook(sync_hook)
        with monkeypatch.context() as m:
            await _run_provider("tavily", m)
        # a failing call reports from the same context too
        with monkeypatch.context() as m:
            _install_http(m, tavily_web_search, {}, status=500)
            with pytest.raises(ValueError):
                await TavilyWebSearchCore().search(QUERY)
    finally:
        marker.reset(token)

    assert seen == ["async:task-42", "sync:task-42", "sync:task-42"]
