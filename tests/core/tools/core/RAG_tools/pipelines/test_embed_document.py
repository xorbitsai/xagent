"""Tests for the extracted embedding stage of document ingestion."""

from __future__ import annotations

import logging
from typing import List, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DatabaseOperationError,
    EmbeddingAdapterError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkForEmbedding,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    EmbeddingStageResult,
    embed_document_chunks,
)

MODEL_ID = "hub-embedding-1"


class _StubAdapter(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: List[Union[str, List[str]]] = []

    def encode(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Union[List[float], List[List[float]]]:
        self.calls.append(text)
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _chunks(count: int, *, start: int = 1) -> List[ChunkForEmbedding]:
    return [
        ChunkForEmbedding(
            doc_id="doc-1",
            chunk_id=f"chunk-{index}",
            parse_hash="hash-1",
            text=f"text-{index}",
            chunk_hash=f"chunk-hash-{index}",
            index=index,
        )
        for index in range(start, start + count)
    ]


def _read_response(chunks: List[ChunkForEmbedding]) -> EmbeddingReadResponse:
    return EmbeddingReadResponse(
        chunks=chunks,
        total_count=len(chunks),
        pending_count=len(chunks),
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    read,
    events: List[tuple],
) -> None:
    monkeypatch.setattr(document_ingestion, "read_chunks_for_embedding", read)

    def _write_vectors(**kwargs: object) -> EmbeddingWriteResponse:
        events.append(
            ("write", kwargs.get("create_index", False), kwargs.get("user_id"))
        )
        embeddings = kwargs.get("embeddings", [])
        return EmbeddingWriteResponse(
            upsert_count=len(embeddings),
            deleted_stale_count=0,
            index_status="created" if kwargs.get("create_index", False) else "skipped",
        )

    def _commit(**kwargs: object) -> None:
        events.append(("commit", kwargs.get("user_id")))

    monkeypatch.setattr(document_ingestion, "write_vectors_to_db", _write_vectors)
    monkeypatch.setattr(document_ingestion, "commit_vectors_to_db", _commit)


def _cfg(*, use_async: bool = False, batch_size: int = 2) -> IngestionConfig:
    return IngestionConfig(
        embedding_use_async=use_async,
        embedding_batch_size=batch_size,
        embedding_concurrent=1,
        max_retries=1,
        retry_delay=0.0,
    )


def _embed(
    adapter: _StubAdapter,
    cfg: IngestionConfig,
    *,
    commit_gate=None,
    result: EmbeddingStageResult | None = None,
    is_spreadsheet: bool = False,
) -> EmbeddingStageResult:
    return embed_document_chunks(
        collection="demo",
        doc_id="doc-1",
        parse_hash="hash-1",
        embedding_config=EmbeddingModelConfig(
            id=MODEL_ID, model_name="stub", dimension=2
        ),
        embedding_adapter=adapter,
        cfg=cfg,
        user_id=42,
        is_admin=True,
        is_spreadsheet=is_spreadsheet,
        commit_gate=commit_gate,
        result=result,
    )


def test_embed_document_chunks_forwards_model_id_and_embeds_returned_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []
    seen: List[dict] = []
    returned = _chunks(1, start=3)

    def _read(**kwargs: object) -> EmbeddingReadResponse:
        seen.append(kwargs)
        return _read_response(returned)

    _install(monkeypatch, _read, events)

    result = _embed(adapter, _cfg(batch_size=10))

    assert seen[0]["model"] == MODEL_ID
    assert adapter.calls == [["text-3"]]
    assert result.embedding_count == 1
    assert result.vector_count == 1
    assert result.pending_count == 1


def test_embed_document_chunks_read_step_metadata_counts_chunks_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return EmbeddingReadResponse(chunks=_chunks(2), total_count=5, pending_count=4)

    _install(monkeypatch, _read, events)

    result = _embed(adapter, _cfg(batch_size=10))

    assert result.read_count == 2
    read_step = document_ingestion._read_chunks_step(result)
    assert read_step.metadata["total_count"] == 2
    assert read_step.metadata["pending_count"] == 4


def test_embed_document_chunks_second_call_computes_nothing_but_still_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []
    responses = [_read_response(_chunks(2)), _read_response([])]

    def _read(**_: object) -> EmbeddingReadResponse:
        return responses.pop(0)

    _install(monkeypatch, _read, events)

    first = _embed(adapter, _cfg())
    calls_after_first = len(adapter.calls)
    second = _embed(adapter, _cfg())

    assert first.embedding_count == 2
    assert second.pending_count == 0
    assert second.embedding_count == 0
    assert second.vector_count == 0
    assert len(adapter.calls) == calls_after_first
    assert [event[0] for event in events] == ["write", "commit", "commit"]


def test_embed_document_chunks_requires_an_owner_for_admin_reads() -> None:
    stage = EmbeddingStageResult()
    with pytest.raises(ValueError, match="document owner"):
        embed_document_chunks(
            collection="demo",
            doc_id="doc-1",
            parse_hash="hash-1",
            embedding_config=EmbeddingModelConfig(
                id=MODEL_ID, model_name="stub", dimension=2
            ),
            embedding_adapter=_StubAdapter(),
            cfg=_cfg(),
            user_id=None,
            is_admin=True,
            is_spreadsheet=False,
            result=stage,
        )

    assert stage.current_step == "read_chunks_for_embedding"


def test_embed_document_chunks_writes_rows_with_the_given_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []
    seen: List[dict] = []

    def _read(**kwargs: object) -> EmbeddingReadResponse:
        seen.append(kwargs)
        return _read_response(_chunks(1))

    _install(monkeypatch, _read, events)

    _embed(adapter, _cfg())

    assert seen[0]["user_id"] == 42
    assert seen[0]["is_admin"] is True
    assert ("write", True, 42) in events
    assert ("commit", 42) in events


def test_embed_document_chunks_calls_commit_gate_before_each_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(3))

    _install(monkeypatch, _read, events)

    def _gate() -> None:
        events.append(("gate",))

    _embed(adapter, _cfg(batch_size=1), commit_gate=_gate)

    assert events == [
        ("gate",),
        ("write", False, 42),
        ("gate",),
        ("write", False, 42),
        ("gate",),
        ("write", True, 42),
        ("commit", 42),
    ]


@pytest.mark.parametrize("use_async", [False, True])
def test_embed_document_chunks_creates_index_only_on_last_batch(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(5))

    _install(monkeypatch, _read, events)

    _embed(adapter, _cfg(use_async=use_async, batch_size=2))

    assert [event[1] for event in events if event[0] == "write"] == [
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("use_async", [False, True])
def test_embed_document_chunks_computes_counts_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(4))

    _install(monkeypatch, _read, events)

    result = _embed(adapter, _cfg(use_async=use_async, batch_size=2))

    assert result.embedding_count == 4
    assert result.vector_count == 4
    assert result.compute_completed is True
    assert result.current_step == "write_vectors_to_db"


def test_embed_document_chunks_logs_compute_completed_before_final_commit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(2))

    def _commit(**_: object) -> None:
        raise RuntimeError("commit failed")

    _install(monkeypatch, _read, events)
    monkeypatch.setattr(document_ingestion, "commit_vectors_to_db", _commit)

    stage = EmbeddingStageResult()
    with caplog.at_level(logging.INFO, logger=document_ingestion.__name__):
        with pytest.raises(RuntimeError, match="commit failed"):
            _embed(adapter, _cfg(batch_size=10), result=stage)

    assert stage.compute_completed is True
    assert any(
        record.message == "Step compute_embeddings completed"
        for record in caplog.records
    )


def test_embed_document_chunks_reports_write_step_and_partial_counts_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(2))

    def _write_vectors(**_: object) -> EmbeddingWriteResponse:
        raise RuntimeError("write failed")

    monkeypatch.setattr(document_ingestion, "read_chunks_for_embedding", _read)
    monkeypatch.setattr(document_ingestion, "write_vectors_to_db", _write_vectors)
    monkeypatch.setattr(
        document_ingestion,
        "commit_vectors_to_db",
        lambda **_: events.append(("commit",)),
    )

    stage = EmbeddingStageResult()
    with pytest.raises(DatabaseOperationError, match="Failed to write embedding batch"):
        _embed(adapter, _cfg(batch_size=10), result=stage)

    assert stage.current_step == "write_vectors_to_db"
    assert stage.embedding_count == 2
    assert stage.vector_count == 0
    assert stage.read_completed is True
    assert stage.compute_completed is False
    assert events == []


def test_embed_document_chunks_async_write_failure_raises_database_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(2))

    def _write_vectors(**_: object) -> EmbeddingWriteResponse:
        raise RuntimeError("write failed")

    monkeypatch.setattr(document_ingestion, "read_chunks_for_embedding", _read)
    monkeypatch.setattr(document_ingestion, "write_vectors_to_db", _write_vectors)
    monkeypatch.setattr(
        document_ingestion,
        "commit_vectors_to_db",
        lambda **_: events.append(("commit",)),
    )

    stage = EmbeddingStageResult()
    with pytest.raises(DatabaseOperationError, match="Failed to write embedding batch"):
        _embed(adapter, _cfg(use_async=True, batch_size=10), result=stage)

    assert stage.current_step == "write_vectors_to_db"
    assert stage.embedding_count == 2
    assert stage.vector_count == 0
    assert stage.compute_completed is False
    assert events == []


@pytest.mark.parametrize(
    ("use_async", "error", "match"),
    [
        (True, EmbeddingAdapterError, "Failed to generate any embeddings"),
        (False, RuntimeError, "provider down"),
    ],
)
def test_embed_document_chunks_fails_before_commit_when_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    error: type[Exception],
    match: str,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return EmbeddingReadResponse(chunks=_chunks(1), total_count=5, pending_count=1)

    def _encode(_text: str) -> List[float]:
        raise RuntimeError("provider down")

    adapter.encode = _encode  # type: ignore[method-assign]
    _install(monkeypatch, _read, events)

    stage = EmbeddingStageResult()
    with pytest.raises(error, match=match) as excinfo:
        _embed(adapter, _cfg(use_async=use_async), result=stage)

    if isinstance(excinfo.value, EmbeddingAdapterError):
        assert excinfo.value.details["chunk_count"] == 5
    assert stage.current_step == "compute_embeddings"
    assert events == []


def test_embed_document_chunks_resets_a_reused_result_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []
    responses = [_read_response(_chunks(2)), _read_response([])]

    def _read(**_: object) -> EmbeddingReadResponse:
        return responses.pop(0)

    _install(monkeypatch, _read, events)

    holder = EmbeddingStageResult()
    _embed(adapter, _cfg(), result=holder)
    _embed(adapter, _cfg(), result=holder)

    assert holder.embedding_count == 0
    assert holder.vector_count == 0
    assert holder.pending_count == 0
    assert holder.compute_completed is False


def test_embed_document_chunks_forces_async_path_for_spreadsheets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    events: List[tuple] = []

    def _read(**_: object) -> EmbeddingReadResponse:
        return _read_response(_chunks(2))

    _install(monkeypatch, _read, events)

    result = _embed(adapter, _cfg(use_async=False, batch_size=10), is_spreadsheet=True)

    assert all(isinstance(call, str) for call in adapter.calls)
    assert result.embedding_count == 2
    assert result.vector_count == 2
