"""Cached collection stats count an ingested document once (real LanceDB)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    IngestionConfig,
    IngestionResult,
)
from xagent.core.tools.core.RAG_tools.management import (
    collection_manager as collection_manager_module,
)
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    get_collection_sync,
)
from xagent.core.tools.core.RAG_tools.management.collections import list_collections
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

COLLECTION = "stats_kb"
CONFIG = EmbeddingModelConfig(
    id="stub-embedding", model_name="stub", model_provider="test", dimension=2
)


class _StubEmbeddingAdapter(BaseEmbedding):
    def encode(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


@pytest.fixture(autouse=True)
def _stub_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _StubEmbeddingAdapter()
    monkeypatch.setattr(
        document_ingestion, "_resolve_embedding_adapter", lambda _c: (CONFIG, adapter)
    )
    monkeypatch.setattr(
        collection_manager_module,
        "resolve_embedding_adapter",
        lambda *_a, **_k: (CONFIG, adapter),
    )


def _ingest(path: Path) -> IngestionResult:
    return document_ingestion.process_document(
        COLLECTION,
        str(path),
        config=IngestionConfig(embedding_model_id=CONFIG.id),
        user_id=1,
        is_admin=True,
    )


def _document(tmp_path: Path, name: str = "a.txt") -> Path:
    path = tmp_path / name
    path.write_text("Milvus is a vector database. " * 5)
    return path


def _stats(info: CollectionInfo) -> tuple[int, int, int, int]:
    return (info.documents, info.parses, info.chunks, info.embeddings)


def _assert_cache_matches_realtime(documents: int) -> None:
    # Read the cache first: a realtime listing writes the fresh stats back to it.
    cached = _stats(get_collection_sync(COLLECTION))
    listing = asyncio.run(
        list_collections(user_id=1, is_admin=True, force_realtime=True)
    )
    (realtime,) = [_stats(c) for c in listing.collections if c.name == COLLECTION]
    assert cached == realtime
    assert cached[0] == documents


def test_one_ingest_counts_the_document_once(tmp_path: Path) -> None:
    assert _ingest(_document(tmp_path)).status == "success"

    _assert_cache_matches_realtime(documents=1)


def test_reingesting_a_document_does_not_count_it_again(tmp_path: Path) -> None:
    path = _document(tmp_path)
    assert _ingest(path).status == "success"
    _assert_cache_matches_realtime(documents=1)

    assert _ingest(path).status == "success"
    _assert_cache_matches_realtime(documents=1)


def test_retry_after_a_failed_attempt_counts_the_document_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _document(tmp_path)
    real_parse = document_ingestion.parse_document

    def _fail_once(*_args: object, **_kwargs: object) -> object:
        monkeypatch.setattr(document_ingestion, "parse_document", real_parse)
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(document_ingestion, "parse_document", _fail_once)
    assert _ingest(path).status != "success"
    assert _ingest(path).status == "success"

    _assert_cache_matches_realtime(documents=1)
