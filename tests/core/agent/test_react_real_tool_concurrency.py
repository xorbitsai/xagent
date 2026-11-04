"""Inc.1 follow-up — the real safe-set tools actually parallelize.

The unit tests elsewhere prove the scheduler's ordering/isolation invariants
with ``FakeTool``. This module instead drives the *real* production tool classes
that are marked ``read_only`` / ``concurrency_safe`` through the real
``ReActPattern`` concurrent path, to guard against two regressions the fakes
cannot catch:

1. A real tool loses its ``concurrency_safe`` metadata (so the scheduler would
   silently fall back to serial) — covered by the segmentation assertions.
2. A real tool's ``run_json_async`` wrapper serializes the batch even though the
   scheduler offered it concurrency (e.g. a future refactor adds blocking work
   around the awaited I/O seam) — covered by the barrier-overlap assertions.

Scope boundary (intentional): the barrier is patched in at each tool's awaited
I/O seam (``fetch_web_content`` / ``*Core.search``), so these tests verify the
scheduler plus the tool's own async wrapper overlap. They do *not* exercise the
production HTTP client itself (which is replaced) — keeping each tool's network
layer non-blocking remains that tool's responsibility, and is asserted only
indirectly via ``run_json_async`` being a coroutine function.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from tests.core.agent.concurrency_harness import (
    ConcurrencyTracker,
    FakeRuntime,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.tools.adapters.vibe.exa_web_search import ExaWebSearchTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.tavily_web_search import TavilyWebSearchTool
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool
from xagent.core.tools.adapters.vibe.zhipu_web_search import ZhipuWebSearchTool

# How long a batched I/O seam waits to rendezvous with its sibling before giving
# up. A serialized batch never rendezvouses, so it trips this instead of hanging.
_RENDEZVOUS_TIMEOUT = 3.0


class _FetchStub:
    """Stand-in for ``WebContentFetchResult`` (only ``as_dict`` is consumed)."""

    def as_dict(self) -> dict[str, Any]:
        return {"success": True, "url": "https://example.com", "content": "ok"}


@dataclass
class _RealToolCase:
    label: str
    tool_factory: Callable[[], Any]
    # Dotted path of the awaited I/O seam to replace with the barrier.
    seam: str
    # Value the patched seam returns once both callers rendezvous.
    seam_return: Any
    args: dict[str, Any]


_CASES = [
    _RealToolCase(
        label="fetch_web_content",
        tool_factory=FetchWebContentTool,
        seam="xagent.core.tools.adapters.vibe.fetch_web_content.fetch_web_content",
        seam_return=_FetchStub(),
        args={"url": "https://example.com"},
    ),
    _RealToolCase(
        label="web_search",
        tool_factory=WebSearchTool,
        seam="xagent.core.tools.adapters.vibe.web_search.WebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
    _RealToolCase(
        label="zhipu_web_search",
        tool_factory=ZhipuWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.zhipu_web_search.ZhipuWebSearchCore.search",
        # Zhipu wraps the response through ``normalize_results``; an empty dict
        # normalizes to an empty result list.
        seam_return={},
        args={"query": "x"},
    ),
    _RealToolCase(
        label="exa_web_search",
        tool_factory=ExaWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.exa_web_search.ExaWebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
    _RealToolCase(
        label="tavily_web_search",
        tool_factory=TavilyWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.tavily_web_search.TavilyWebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
]

_CASE_IDS = [case.label for case in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_real_safe_tool_is_batched_as_concurrent(case: _RealToolCase) -> None:
    """Two calls to a real safe tool are recognized as a concurrent segment."""
    tool = case.tool_factory()
    pattern = make_react(parallel=True, max_concurrency=3)
    # The registered tool name can differ from the test label (e.g. the Google
    # and Tavily backends both expose "web_search"); drive the scheduler with
    # the real name so ``_find_tool`` resolves the metadata.
    batch = [make_tool_call(tool.name, case.args) for _ in range(2)]

    segment, kind = pattern._next_segment(batch, [tool])

    assert kind == "concurrent"
    assert len(segment) == 2


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_real_safe_tool_exposes_async_executor(case: _RealToolCase) -> None:
    """The real tool's executor is a coroutine function (yields to the loop)."""
    tool = case.tool_factory()
    assert inspect.iscoroutinefunction(tool.run_json_async)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
async def test_real_safe_tool_runs_concurrently(
    case: _RealToolCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real tool's async path overlaps under the concurrent scheduler.

    A two-party barrier sits at the tool's awaited I/O seam: it only releases
    once both calls are in flight, so a passing run proves real overlap (peak
    concurrency == 2) and ordered, one-result-per-call backfill. A serialized
    run never reaches the barrier together and trips ``_RENDEZVOUS_TIMEOUT``,
    surfacing as timed-out (failed) results rather than a hang.
    """
    tracker = ConcurrencyTracker()
    barrier = asyncio.Barrier(2)

    async def _seam(*_args: Any, **_kwargs: Any) -> Any:
        tracker.enter(case.label)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=_RENDEZVOUS_TIMEOUT)
        finally:
            tracker.leave(case.label)
        return case.seam_return

    monkeypatch.setattr(case.seam, _seam)

    tool = case.tool_factory()
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(tool.name, case.args) for _ in range(2)]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    # Real overlap: both calls were in flight at once.
    assert tracker.peak == 2
    # I1/I2: one result per call, in input order, none of them a seam timeout.
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]
