"""Every built-in web search provider sends only the redacted query.

HTTP uses a real ``httpx`` client on ``httpx.MockTransport``; the Exa SDK call
is stubbed. All personal data below is invented.
"""

import copy
import importlib
import json
import logging
import pkgutil
import sys
import types
from typing import Any, Dict, List

import httpx
import pytest

from xagent.core.agent.trace import (
    SEARCH_TOOL_NAMES,
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
async def test_cancelled_search_is_reported_and_stays_cancelled(monkeypatch):
    import asyncio

    events: List[SearchAuditEvent] = []
    set_search_audit_hook(events.append)
    started = asyncio.Event()

    async def slow(_q: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    from xagent.core.tools.core.web_search_guard import run_guarded_search

    task = asyncio.ensure_future(run_guarded_search("google", QUERY, slow, len))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].failure_kind == "cancelled"
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


# --- structural projection of search calls in console payloads ----------------

from xagent.core.agent.trace import _render_event_data_for_log  # noqa: E402


def _args(query: str = QUERY) -> Dict[str, Any]:
    return {"query": query, "num_results": 3}


def _real_shapes(query: str = QUERY) -> Dict[str, Dict[str, Any]]:
    openai_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps(_args(query))},
    }
    return {
        # run_streaming_llm_call -> on_llm_end: normalised {name, args} calls
        "llm_end_args": {
            "response": {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "web_search", "args": _args(query)}
                ],
            },
            "success": True,
        },
        # raw OpenAI shape: arguments is a JSON string
        "llm_end_openai": {
            "response": {"content": "", "tool_calls": [openai_call]},
            "success": True,
        },
        # the prompt replay in an LLM start event
        "context_preview": {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [openai_call]},
            ]
        },
        # checkpoint snapshot
        "checkpoint": {
            "pattern_state": {
                "pending_tool_calls": [
                    {"id": "c1", "name": "web_search", "args": _args(query)}
                ],
                "tool_ledger": [
                    {"tool_name": "exa_web_search", "tool_params": _args(query)}
                ],
                "last_response": {"tool_calls": [openai_call]},
            }
        },
    }


@pytest.mark.parametrize("shape", list(_real_shapes()))
@pytest.mark.parametrize("max_bytes", [None, 0])
def test_console_render_redacts_real_payload_shapes(shape, max_bytes):
    data = _real_shapes()[shape]
    snapshot = copy.deepcopy(data)
    rendered = _render_event_data_for_log(data, max_bytes)
    for value in RAW_VALUES:
        assert value not in rendered
    assert "[EMAIL]" in rendered
    # the stored payload is untouched
    assert data == snapshot


def test_console_render_redacts_unparseable_arguments_string():
    data = {
        "function": {
            "name": "web_search",
            "arguments": '{"query": "mail ' + EMAIL + ' now", "num_res',
        }
    }
    rendered = _render_event_data_for_log(data)
    assert EMAIL not in rendered and "mail" not in rendered
    assert "[search arguments withheld]" in rendered


@pytest.mark.parametrize("shape", list(_real_shapes()))
def test_no_fragment_of_the_query_survives_any_byte_budget(shape):
    # Redaction happens before the byte budget cuts the payload, so no budget
    # can leave the start of an identifier behind.
    pad = "x " * 200
    data = _real_shapes(pad + QUERY)[shape]
    full = len(_render_event_data_for_log(data, 10**9))
    fragments = ["jane", "example.com", "412 3", "Smith", "123 4", "1234567"]
    for budget in range(40, full + 40, 13):
        rendered = _render_event_data_for_log(data, budget)
        for fragment in fragments:
            assert fragment not in rendered, (budget, fragment)


def _replay_message(shape: str, query: str) -> Dict[str, Any]:
    arguments = json.dumps({"query": query, "n": 3})
    return {
        "openai_str": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": arguments},
                }
            ],
        },
        "arguments_dict": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"query": query}},
                }
            ],
        },
        # string arguments inside a dict that _shrink_message dumps again
        "legacy_function_call": {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "web_search", "arguments": arguments},
        },
        "content_blocks": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "web_search",
                    "input": {"query": query},
                }
            ],
        },
        "name_args": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "web_search", "args": {"query": query}}
            ],
        },
    }[shape]


@pytest.mark.parametrize(
    "shape",
    [
        "openai_str",
        "arguments_dict",
        "legacy_function_call",
        "content_blocks",
        "name_args",
    ],
)
@pytest.mark.parametrize("pad", [0, 3, 8, 14])
def test_normalizer_then_console_never_leaves_a_query_fragment(shape, pad):
    # The trace boundary may cut a search call's arguments before the console
    # sink sees them (normalize_llm_trace_payload). Whatever budget cuts them,
    # the console line must carry no part of the identifier.
    from xagent.core.agent.trace import normalize_llm_trace_payload

    query = "word " * pad + "find jane.doe@example.com now"
    messages = [
        {"role": "user", "content": "hi " + "x" * 300},
        _replay_message(shape, query),
    ]
    fragments = ["jane", "@exa", "doe@", "example"]
    for budget in range(250, 1500, 3):
        stored = normalize_llm_trace_payload(
            {"context_preview": copy.deepcopy(messages), "success": True},
            max_bytes=budget,
        )
        snapshot = copy.deepcopy(stored)
        rendered = _render_event_data_for_log(stored)
        assert stored == snapshot  # the stored payload is untouched
        for fragment in fragments:
            assert fragment not in rendered, (shape, pad, budget, fragment)


def test_unparsed_arguments_fallback_input_is_redacted():
    # ReAct passes malformed JSON arguments on as {"input": <raw text>}.
    data = {
        "tool_name": "web_search",
        "tool_params": {"input": f"mail {EMAIL} or call {PHONE}"},
    }
    snapshot = copy.deepcopy(data)
    for max_bytes in (None, 0):
        rendered = _render_event_data_for_log(data, max_bytes)
        assert EMAIL not in rendered and "412 345" not in rendered
        assert "[EMAIL]" in rendered and "[PHONE]" in rendered
    assert data == snapshot


def test_every_string_leaf_of_search_arguments_is_redacted():
    data = {
        "tool_name": "web_search",
        "tool_params": {
            "query": "ok",
            "filters": {"site": EMAIL, "tags": [f"a {PHONE}"]},
            "num_results": 3,
        },
    }
    rendered = _render_event_data_for_log(data)
    assert EMAIL not in rendered and "412 345" not in rendered
    assert "'num_results': 3" in rendered


@pytest.mark.parametrize(
    "text",
    [
        '{"query": "find jane.doe@exa...[truncated 20 chars]',
        "{'query': 'find jane.doe@exa...[truncated 20 chars]",
        '[{"function_call": "{\\"query\\": \\"find jane.doe@exa...[truncated 9 chars]',
        '{\\\\"query\\\\" : \\\\"jane.doe@exa...[truncated 9 chars]',
    ],
)
def test_truncated_text_holding_a_query_key_is_withheld_whole(text):
    rendered = _render_event_data_for_log({"content": text})
    assert "jane" not in rendered and "@exa" not in rendered
    assert "[truncated text containing a query withheld]" in rendered


def test_cut_leaf_inside_search_arguments_is_withheld():
    data = {
        "tool_name": "web_search",
        "tool_params": {"query": "find jane.doe@exa...[truncated 9 chars]"},
    }
    for max_bytes in (None, 0):
        rendered = _render_event_data_for_log(data, max_bytes)
        assert "jane" not in rendered and "@exa" not in rendered
        assert "[search arguments withheld]" in rendered


def test_cut_query_text_is_withheld_when_nothing_is_truncated_by_the_sink():
    data = {"content": '{"query": "jane.doe@exa...[truncated 9 chars]'}
    rendered = _render_event_data_for_log(data, 0)
    assert "jane" not in rendered
    assert "[truncated text containing a query withheld]" in rendered


def test_stringified_block_naming_a_search_tool_without_a_query_key():
    # the {"input": <raw text>} fallback inside a stringified content block
    complete = json.dumps(
        [{"type": "tool_use", "name": "web_search", "input": {"input": EMAIL}}]
    )
    rendered = _render_event_data_for_log({"content": complete})
    assert EMAIL not in rendered and "[EMAIL]" in rendered
    cut = complete[:-12] + "...[truncated 12 chars]"
    rendered = _render_event_data_for_log({"content": cut})
    assert "jane" not in rendered
    assert "[truncated text containing a query withheld]" in rendered


def test_unquoted_search_tool_name_in_prose_is_kept():
    text = "I will use web_search for this...[truncated 40 chars]"
    assert text in _render_event_data_for_log({"content": text})


def test_truncated_text_without_a_query_key_is_kept():
    text = "an ordinary long answer...[truncated 99 chars]"
    assert text in _render_event_data_for_log({"content": text})


def test_truncation_marker_constant_matches_every_reducer():
    from xagent.core.agent import trace

    marker = trace._TRUNCATION_MARKER
    assert marker in trace._truncate_string("x" * 500, 50)
    assert marker in trace._trim_subtree("x" * 500, 50)
    assert marker in trace._shrink_node("x" * 500, [100], 1, set(), 12)
    middle = trace._reduce_text_list(["x" * 200] * 20, 400)
    assert any(isinstance(i, str) and marker in i for i in middle)


def test_console_render_leaves_other_tools_alone():
    data = {"tool_calls": [{"name": "write_file", "args": {"query": EMAIL}}]}
    assert EMAIL in _render_event_data_for_log(data)


def test_search_tool_name_set_covers_every_registered_search_tool():
    import xagent.core.tools.adapters.vibe as vibe

    names = set()
    for mod in pkgutil.iter_modules(vibe.__path__):
        if not mod.name.endswith("web_search"):
            continue
        module = importlib.import_module(f"{vibe.__name__}.{mod.name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and obj.__name__.endswith("Tool")
            ):
                names.add(obj.name.fget(None))
    assert names, "no search tool classes discovered"
    assert names <= SEARCH_TOOL_NAMES, names - SEARCH_TOOL_NAMES


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
