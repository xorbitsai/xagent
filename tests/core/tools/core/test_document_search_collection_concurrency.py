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


def _install_collections(
    monkeypatch: pytest.MonkeyPatch, *names: str, embeddings: int = 10
) -> None:
    async def _list(
        user_id: int | None = None,
        is_admin: bool = False,
        governing_team_id: int | None = None,
    ) -> ListCollectionsResult:
        del user_id, is_admin, governing_team_id
        listing = _collections(*names)
        for collection in listing.collections:
            collection.embeddings = embeddings
        return listing

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
        # Passing the barrier needs three OS threads resident at once, which is
        # only possible if the loop dispatched all three without blocking on any.
        barrier.wait()
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", blocking_search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert len(result.results) == 3
    assert loop_thread_id not in worker_thread_ids


@pytest.mark.asyncio
async def test_search_runs_readonly_so_retrieval_never_builds_an_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent retrieval must not reach create_index's LanceDB commit."""
    _install_collections(monkeypatch, "alpha", "beta")
    configs: list[dict[str, Any]] = []

    def search(
        *, collection: str, config: dict[str, Any], **_kwargs: Any
    ) -> SearchPipelineResult:
        configs.append(config)
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert len(configs) == 2
    assert all(config["readonly"] is True for config in configs)


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
            # Empty message so the warnings list itself has to reach the summary.
            return _pipeline_result(
                collection,
                status="partial_success",
                message="",
                warnings=["fell back to sparse"],
            )
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert [entry.collection for entry in result.results] == ["alpha", "warned"]
    # Split the sections apart so a failure cannot pass as a mere warning.
    warnings_section, errors_section = result.summary.split("\n\nErrors: ")
    warnings_section = warnings_section.split("\n\nWarnings: ")[1]
    assert errors_section == "boom: connection reset | status_error: nope"
    assert warnings_section == "warned: fell back to sparse"


@pytest.mark.asyncio
async def test_aggregation_order_totals_and_empty_collections_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order follows the collection list and zero-embedding collections are skipped."""

    async def _list(
        user_id: int | None = None,
        is_admin: bool = False,
        governing_team_id: int | None = None,
    ) -> ListCollectionsResult:
        del user_id, is_admin, governing_team_id
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


@pytest.mark.asyncio
async def test_every_collection_filtered_out_fans_out_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-empty listing must gather zero tasks, not fail on the empty fan-out."""
    _install_collections(monkeypatch, "alpha", "beta", embeddings=0)

    def search(**_kwargs: Any) -> SearchPipelineResult:
        raise AssertionError("collections without embeddings must not be searched")

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert result.results == []
    assert result.summary.startswith("No relevant documents found")
    assert "Searched 0 documents" in result.summary


@pytest.mark.asyncio
async def test_all_collections_failing_returns_the_errors_only_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no survivor the summary is the errors branch, not the no-results one."""
    _install_collections(monkeypatch, "alpha", "beta")

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        raise RuntimeError(f"{collection} is down")

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert result.results == []
    assert result.summary.startswith("Knowledge base search failed for one or more")
    assert "alpha: alpha is down" in result.summary
    assert "beta: beta is down" in result.summary
    assert "No relevant documents found" not in result.summary


@pytest.mark.asyncio
async def test_real_warnings_are_not_masked_by_the_boilerplate_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """readonly=True makes every hybrid search partial_success with a generic
    message; the actual diagnostics must still reach the summary."""
    _install_collections(monkeypatch, "alpha")

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        return _pipeline_result(
            collection,
            status="partial_success",
            # Exactly what the hybrid pipeline sets alongside its warnings.
            message="Hybrid search completed with warnings",
            warnings=[
                "READONLY_MODE: Readonly mode enabled for sparse search on model.",
                "FTS_INDEX_MISSING: FTS index not found on 'text' column for model.",
            ],
        )

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert "alpha: FTS_INDEX_MISSING" in result.summary
    assert "Hybrid search completed with warnings" not in result.summary
    # Our own readonly flag is not a diagnostic worth showing the agent.
    assert "READONLY_MODE" not in result.summary


@pytest.mark.asyncio
async def test_the_self_inflicted_readonly_notice_produces_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean readonly search must not decorate its summary with a warning."""
    _install_collections(monkeypatch, "alpha")

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        return _pipeline_result(
            collection,
            status="partial_success",
            message="Hybrid search completed with warnings",
            warnings=["READONLY_MODE: Readonly mode enabled for sparse search."],
        )

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert len(result.results) == 1
    assert "Warnings:" not in result.summary


@pytest.mark.asyncio
async def test_warnings_survive_when_no_collection_returns_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warning from an empty-result collection still reaches the summary."""
    _install_collections(monkeypatch, "alpha")

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        return _pipeline_result(
            collection,
            status="partial_success",
            results=[],
            result_count=0,
            message="",
            warnings=["index still building"],
        )

    monkeypatch.setattr(document_search, "run_document_search", search)

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    assert result.results == []
    assert result.summary.startswith("No relevant documents found")
    assert "Warnings: alpha: index still building" in result.summary


@pytest.mark.asyncio
async def test_a_broken_collection_object_cannot_escape_into_the_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_search_one promises never to raise, including on its attribute reads.

    gather() has no return_exceptions, so an attribute read that escaped would
    abort the whole batch and leave the sibling tasks running uncancelled.
    """

    class _Exploding:
        name = "broken"
        embeddings = 5
        documents = 3

        @property
        def rerank_model_id(self) -> str:
            # getattr's default only swallows AttributeError, not this.
            raise RuntimeError("rerank binding is corrupt")

    async def _list(
        user_id: int | None = None,
        is_admin: bool = False,
        governing_team_id: int | None = None,
    ) -> ListCollectionsResult:
        del user_id, is_admin, governing_team_id
        listing = _collections("alpha")
        listing.collections.append(_Exploding())  # type: ignore[arg-type]
        return listing

    monkeypatch.setattr(document_search, "_list_visible_collections", _list)
    monkeypatch.setattr(
        document_search,
        "run_document_search",
        lambda *, collection, **_kwargs: _pipeline_result(collection),
    )

    result = await document_search._search_knowledge_base_impl(_args(), user_id=1)

    # The healthy sibling still returns; the broken one degrades to an error.
    assert [entry.collection for entry in result.results] == ["alpha"]
    assert "broken: rerank binding is corrupt" in result.summary


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_a_stuck_collection_times_out_without_holding_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-collection deadline bounds the fan-out and spares its siblings."""
    _install_collections(monkeypatch, "stuck", "fast")
    monkeypatch.setattr(document_search, "get_kb_search_timeout_seconds", lambda: 0.2)
    release = threading.Event()

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        if collection == "stuck":
            release.wait()
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    try:
        result = await document_search._search_knowledge_base_impl(_args(), user_id=1)
    finally:
        release.set()
    assert [entry.collection for entry in result.results] == ["fast"]
    assert "stuck: search timed out after 0.2s" in result.summary


@pytest.mark.asyncio
async def test_cancelling_the_fan_out_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the caller must cancel the gather, not surface a partial result."""
    _install_collections(monkeypatch, "alpha", "beta")
    started = threading.Event()
    release = threading.Event()

    def search(*, collection: str, **_kwargs: Any) -> SearchPipelineResult:
        started.set()
        release.wait(10)
        return _pipeline_result(collection)

    monkeypatch.setattr(document_search, "run_document_search", search)

    task = asyncio.create_task(
        document_search._search_knowledge_base_impl(_args(), user_id=1)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
