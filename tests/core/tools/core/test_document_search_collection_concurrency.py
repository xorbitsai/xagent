"""Regression coverage for concurrent multi-collection knowledge base search."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from xagent.core.tools.core import document_search
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
    SearchPipelineResult,
)


def _collections(*names: str) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=name, embeddings=10, documents=3) for name in names
        ],
        total_count=len(names),
        message="ok",
    )


def _pipeline_result(collection: str, **overrides: Any) -> SearchPipelineResult:
    payload: dict[str, Any] = {
        "status": "success",
        "search_type": "dense",
        "results": [
            {
                "doc_id": f"{collection}-doc",
                "chunk_id": f"{collection}-chunk",
                "text": f"hit from {collection}",
                "score": 0.9,
                "parse_hash": "hash",
                "model_tag": "model",
                "metadata": {"source": f"/kb/{collection}.md"},
            }
        ],
        "result_count": 1,
        "message": "ok",
    }
    payload.update(overrides)
    return SearchPipelineResult(**payload)


def _install_collections(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    async def _list(
        user_id: int | None = None, is_admin: bool = False
    ) -> ListCollectionsResult:
        del user_id, is_admin
        return _collections(*names)

    monkeypatch.setattr(document_search, "_list_visible_collections", _list)


def _args(**overrides: Any) -> document_search.KnowledgeSearchArgs:
    payload: dict[str, Any] = {"query": "what is xagent"}
    payload.update(overrides)
    return document_search.KnowledgeSearchArgs(**payload)


@pytest.mark.asyncio
async def test_collections_are_searched_concurrently_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three collections must overlap in flight, none of them on the loop thread."""
    _install_collections(monkeypatch, "alpha", "beta", "gamma")
    loop_thread_id = threading.get_ident()
    barrier = threading.Barrier(3, timeout=5)
    worker_thread_ids: set[int] = set()
    lock = threading.Lock()

    def blocking_search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        with lock:
            worker_thread_ids.add(threading.get_ident())
        # Only passes if all three searches are simultaneously in flight.
        barrier.wait()
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", blocking_search)

    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker = asyncio.create_task(tick())
    try:
        result = await document_search._search_knowledge_base_impl(_args(), user_id=1)
    finally:
        ticker.cancel()

    assert len(result.results) == 3
    # The loop stayed responsive while the blocking pipeline ran.
    assert ticks > 0
    assert loop_thread_id not in worker_thread_ids


@pytest.mark.asyncio
async def test_one_failing_collection_does_not_drop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises, error status and warnings stay per-collection."""
    _install_collections(monkeypatch, "alpha", "boom", "status_error", "warned")

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        if collection == "boom":
            raise RuntimeError("connection reset")
        if collection == "status_error":
            return _pipeline_result(
                collection, status="error", results=[], result_count=0, message="nope"
            )
        if collection == "warned":
            return _pipeline_result(
                collection, status="partial_success", warnings=["fell back to vector"]
            )
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert [entry.collection for entry in result.results] == ["alpha", "warned"]
    assert "boom: connection reset" in result.summary
    assert "status_error: nope" in result.summary
    assert "warned: ok" in result.summary


@pytest.mark.asyncio
async def test_aggregation_order_totals_and_empty_collections_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order follows the collection list and zero-embedding collections are skipped."""
    _install_collections(monkeypatch, "alpha", "empty", "beta")

    async def _list(
        user_id: int | None = None, is_admin: bool = False
    ) -> ListCollectionsResult:
        del user_id, is_admin
        listing = _collections("alpha", "empty", "beta")
        listing.collections[1].embeddings = 0
        return listing

    monkeypatch.setattr(document_search, "_list_visible_collections", _list)

    searched: list[str] = []
    delays = {"alpha": 0.15, "beta": 0.0}

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        searched.append(collection)
        time.sleep(delays[collection])
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert sorted(searched) == ["alpha", "beta"]
    # Slow "alpha" finishes last but still aggregates first.
    assert [entry.collection for entry in result.results] == ["alpha", "beta"]
    # total_searched counts documents of the two non-empty collections only.
    assert result.summary.startswith("Found 2 relevant results from 6 documents")
