"""Unit tests for the document search pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.storage import initialize_storage_manager
from xagent.core.tools.core.RAG_tools.chunk.chunk_document import chunk_document
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    ParseMethod,
    SearchConfig,
    SearchPipelineResult,
    SearchResult,
    SearchType,
)
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document
from xagent.core.tools.core.RAG_tools.pipelines import document_search
from xagent.core.tools.core.RAG_tools.pipelines.document_search import (
    _apply_rerank_if_needed,
    _resolve_dashscope_rerank,
)
from xagent.core.tools.core.RAG_tools.vector_storage import index_manager as idx_module
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    write_vectors_to_db,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


class _FakeEmbeddingAdapter(BaseEmbedding):
    """Local embedding adapter that deterministically encodes text length."""

    def encode(
        self, text: Any, dimension: int | None = None, instruct: str | None = None
    ) -> Any:
        if isinstance(text, str):
            return [float(len(text))]
        return [[float(len(item))] for item in text]

    def get_dimension(self) -> int:
        return 1

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _patch_embedding_adapter(
    monkeypatch: pytest.MonkeyPatch, embedding_model_id: str
) -> None:
    """Force search pipeline to use local fake embedding adapter."""

    fake_config = type(
        "FakeConfig",
        (),
        {"id": embedding_model_id, "model_name": embedding_model_id},
    )()

    monkeypatch.setattr(
        document_search,
        "resolve_embedding_adapter",
        lambda _model_id, api_key=None, base_url=None, timeout_sec=None: (
            fake_config,
            _FakeEmbeddingAdapter(),
        ),
    )


@pytest.mark.integration
def test_document_search_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the full document pipeline on a real PDF and ensure sparse search works."""

    # -------- Environment bootstrap --------
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))

    storage_root = tmp_path / "storage"
    uploads_dir = storage_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    initialize_storage_manager(str(storage_root), str(uploads_dir))

    embedding_model_id = "rag-test-embedding"
    _patch_embedding_adapter(monkeypatch, embedding_model_id)

    # Mock collection manager for embedding binding
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(
        name="test_collection",
        embedding_model_id=embedding_model_id,
        embedding_dimension=384,  # Fake dimension
    )

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_collection_sync",
        lambda collection_name: (
            mock_collection
            if collection_name == "test_collection"
            else CollectionInfo(name=collection_name)
        ),
    )
    # Ensure index manager creates FTS indices
    idx_policy = idx_module.IndexPolicy(fts_enabled=True)
    idx_instance = idx_module.IndexManager(idx_policy)
    monkeypatch.setattr(
        idx_module, "_default_index_manager", idx_instance, raising=False
    )
    monkeypatch.setattr(
        idx_module,
        "get_index_manager",
        lambda policy=None: idx_instance,
    )

    # -------- Pipeline execution --------
    tests_root = Path(__file__).resolve().parents[5]
    test_pdf = tests_root / "resources" / "test_files" / "test.pdf"
    collection = f"collection_{uuid.uuid4().hex[:8]}"
    doc_id = uuid.uuid4().hex

    register_document(
        collection=collection,
        source_path=str(test_pdf),
        doc_id=doc_id,
        user_id=1,
    )

    parse_result = parse_document(
        collection=collection,
        doc_id=doc_id,
        parse_method=ParseMethod.PYPDF,
        user_id=1,
        is_admin=True,
    )
    parse_hash = parse_result["parse_hash"]

    chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        user_id=1,
    )

    embedding_read = read_chunks_for_embedding(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        model=embedding_model_id,
        user_id=1,
    )
    assert embedding_read.chunks, "Expected parsed chunks for embedding"

    embeddings: List[ChunkEmbeddingData] = []
    for chunk in embedding_read.chunks:
        text = chunk.text or ""
        embeddings.append(
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=embedding_model_id,
                vector=[float(len(text))],
                text=text,
                chunk_hash=chunk.chunk_hash,
            )
        )

    write_vectors_to_db(
        collection=collection,
        embeddings=embeddings,
        create_index=True,
    )

    # -------- Execute sparse search --------
    first_chunk_text = embedding_read.chunks[0].text.strip()
    assert first_chunk_text, "First chunk should contain textual content"
    query_text = first_chunk_text.split()[0]
    search_result = document_search.search_documents(
        collection=collection,
        query_text=query_text,
        config=SearchConfig(
            search_type=SearchType.SPARSE,
            top_k=3,
            embedding_model_id=embedding_model_id,
        ),
    )

    assert isinstance(search_result, SearchPipelineResult)
    assert search_result.status == "success"
    assert search_result.result_count > 0
    assert any(
        query_text.lower() in result.text.lower() for result in search_result.results
    )

    # FTS index should have been created without config errors
    conn = get_connection_from_env()
    table = conn.open_table(f"embeddings_{to_model_tag(embedding_model_id)}")
    assert idx_instance.get_fts_index_status(table) is True


@pytest.mark.integration
def test_chinese_sparse_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sparse search with Chinese text content and queries."""

    # -------- Environment bootstrap --------
    lancedb_dir = tmp_path / "lancedb_chinese"
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))

    storage_root = tmp_path / "storage_chinese"
    uploads_dir = storage_root / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    initialize_storage_manager(str(storage_root), str(uploads_dir))

    embedding_model_id = "rag-test-embedding-chinese"
    _patch_embedding_adapter(monkeypatch, embedding_model_id)

    # Mock collection manager for embedding binding
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(
        name="test_collection_chinese",
        embedding_model_id=embedding_model_id,
        embedding_dimension=384,
    )

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.get_collection_sync",
        lambda collection_name: (
            mock_collection
            if collection_name == "test_collection_chinese"
            else CollectionInfo(name=collection_name)
        ),
    )

    # Ensure index manager creates FTS indices
    idx_policy = idx_module.IndexPolicy(fts_enabled=True)
    idx_instance = idx_module.IndexManager(idx_policy)
    monkeypatch.setattr(
        idx_module, "_default_index_manager", idx_instance, raising=False
    )
    monkeypatch.setattr(
        idx_module,
        "get_index_manager",
        lambda policy=None: idx_instance,
    )

    # -------- Create Chinese test document --------
    tests_root = Path(__file__).resolve().parents[5]
    test_txt = tests_root / "resources" / "test_files" / "test.txt"

    # Check if test.txt exists and contains Chinese
    if not test_txt.exists():
        pytest.skip(f"Test file not found: {test_txt}")

    # Create a temporary Chinese text file if needed
    chinese_test_file = tmp_path / "chinese_test.txt"
    chinese_content = """人工智能是计算机科学的一个分支。
机器学习是人工智能的核心技术之一。
自然语言处理用于理解和生成人类语言。
深度学习使用神经网络进行学习。
这些技术正在改变我们的世界。"""
    chinese_test_file.write_text(chinese_content, encoding="utf-8")

    collection = f"collection_chinese_{uuid.uuid4().hex[:8]}"
    doc_id = uuid.uuid4().hex

    # Use the Chinese test file
    register_document(
        collection=collection,
        source_path=str(chinese_test_file),
        doc_id=doc_id,
        user_id=1,
    )

    parse_result = parse_document(
        collection=collection,
        doc_id=doc_id,
        parse_method=ParseMethod.DEEPDOC,
        user_id=1,
        is_admin=True,
    )
    parse_hash = parse_result["parse_hash"]

    chunk_document(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        user_id=1,
    )

    embedding_read = read_chunks_for_embedding(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        model=embedding_model_id,
        user_id=1,
    )
    assert embedding_read.chunks, "Expected parsed chunks for embedding"

    # Check if chunks contain Chinese text
    has_chinese = any(
        any("\u4e00" <= char <= "\u9fff" for char in chunk.text)
        for chunk in embedding_read.chunks
    )
    if not has_chinese:
        pytest.skip("No Chinese content found in parsed chunks")

    embeddings: List[ChunkEmbeddingData] = []
    for chunk in embedding_read.chunks:
        text = chunk.text or ""
        embeddings.append(
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=embedding_model_id,
                vector=[float(len(text))],
                text=text,
                chunk_hash=chunk.chunk_hash,
            )
        )

    write_vectors_to_db(
        collection=collection,
        embeddings=embeddings,
        create_index=True,
    )

    # -------- Test Chinese sparse search queries --------
    chinese_queries = [
        "人工智能",
        "机器学习",
        "自然语言处理",
        "深度学习",
    ]

    for query_text in chinese_queries:
        search_result = document_search.search_documents(
            collection=collection,
            query_text=query_text,
            config=SearchConfig(
                search_type=SearchType.SPARSE,
                top_k=5,
                embedding_model_id=embedding_model_id,
            ),
        )

        assert isinstance(search_result, SearchPipelineResult)
        print(f"\n查询: {query_text}")
        print(f"状态: {search_result.status}")
        print(f"结果数量: {search_result.result_count}")
        # SearchPipelineResult.warnings is List[str], not List[SearchWarning]
        fts_warning_present = any("FTS" in w for w in search_result.warnings)
        print(f"FTS 启用: {not fts_warning_present}")

        if search_result.warnings:
            for warning in search_result.warnings:
                print(f"  警告: {warning}")

        # Check if FTS worked or fell back to substring search
        if search_result.result_count > 0:
            print(f"  找到 {search_result.result_count} 个结果")
            for i, result in enumerate(search_result.results[:3], 1):
                print(f"    {i}. 分数: {result.score:.4f}, 文本: {result.text[:50]}...")
            # Verify that query text appears in results (either via FTS or substring fallback)
            assert any(query_text in result.text for result in search_result.results), (
                f"Query '{query_text}' should appear in search results"
            )
        else:
            print("  未找到结果（可能 FTS 不支持中文分词）")

        # Check for FTS fallback warning
        # SearchPipelineResult.warnings is List[str], not List[SearchWarning]
        has_fallback_warning = any("FTS_FALLBACK" in w for w in search_result.warnings)
        if has_fallback_warning:
            print("  ⚠️  使用了子串匹配回退（FTS 可能不支持中文分词）")

    # Verify FTS index status
    conn = get_connection_from_env()
    table = conn.open_table(f"embeddings_{to_model_tag(embedding_model_id)}")
    fts_enabled = idx_instance.get_fts_index_status(table)
    print(f"\nFTS 索引状态: {fts_enabled}")
    print("=" * 60)


def test_resolve_dashscope_rerank_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test environment variable priority for DashScope rerank configuration."""
    # Test 1: DASHSCOPE_RERANK_* variables are used (no fallback to DASHSCOPE_MODEL)
    monkeypatch.setenv("DASHSCOPE_RERANK_ENABLED", "true")
    monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shared-key")

    rerank = _resolve_dashscope_rerank()
    assert rerank is not None
    assert rerank.model == "qwen3-rerank"
    assert rerank.api_key == "rerank-key"

    # Test 2: Fallback to DASHSCOPE_API_KEY when DASHSCOPE_RERANK_API_KEY not set
    monkeypatch.delenv("DASHSCOPE_RERANK_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "shared-key")
    monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")

    rerank = _resolve_dashscope_rerank()
    assert rerank is not None
    assert rerank.model == "qwen3-rerank"
    assert rerank.api_key == "shared-key"  # Should fallback to shared key

    # Test 3: Disabled when DASHSCOPE_RERANK_ENABLED=false
    monkeypatch.setenv("DASHSCOPE_RERANK_ENABLED", "false")
    rerank = _resolve_dashscope_rerank()
    assert rerank is None

    # Test 4: Returns None when API key missing
    monkeypatch.setenv("DASHSCOPE_RERANK_ENABLED", "true")
    monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
    monkeypatch.delenv("DASHSCOPE_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    rerank = _resolve_dashscope_rerank()
    assert rerank is None


def test_apply_rerank_dashscope_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DashScope rerank succeeds."""
    # Setup mock DashScope rerank
    mock_rerank = MagicMock()
    mock_rerank.compress.return_value = ["doc3", "doc1", "doc2"]

    # Create test results
    results = [
        SearchResult(
            doc_id="d1",
            chunk_id="c1",
            text="doc1",
            score=0.8,
            parse_hash="ph1",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.8,
            fts_score=0.7,
            vector_rank=1,
            fts_rank=2,
        ),
        SearchResult(
            doc_id="d2",
            chunk_id="c2",
            text="doc2",
            score=0.7,
            parse_hash="ph2",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.7,
            fts_score=0.8,
            vector_rank=2,
            fts_rank=1,
        ),
        SearchResult(
            doc_id="d3",
            chunk_id="c3",
            text="doc3",
            score=0.6,
            parse_hash="ph3",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.6,
            fts_score=0.6,
            vector_rank=3,
            fts_rank=3,
        ),
    ]

    cfg = SearchConfig(embedding_model_id="test-embed", rerank_top_k=5)

    # Mock _resolve_dashscope_rerank to return our mock
    with patch(
        "xagent.core.tools.core.RAG_tools.pipelines.document_search._resolve_dashscope_rerank",
        return_value=mock_rerank,
    ):
        reranked, used_rerank, warnings = _apply_rerank_if_needed(
            results, "test query", cfg
        )

    assert used_rerank is True
    assert len(reranked) == 3
    assert reranked[0].text == "doc3"
    assert reranked[1].text == "doc1"
    assert reranked[2].text == "doc2"
    assert len(warnings) == 0
    mock_rerank.compress.assert_called_once_with(["doc1", "doc2", "doc3"], "test query")


def test_apply_rerank_dashscope_failure_fallback_to_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test DashScope rerank failure falls back to LanceDB RRF."""
    # Setup mock DashScope rerank that raises exception
    # Use RequestException to match actual DashScope API behavior
    mock_rerank = MagicMock()
    mock_rerank.compress.side_effect = requests.exceptions.RequestException(
        "DashScope API error"
    )

    # Create test results with original scores/ranks
    results = [
        SearchResult(
            doc_id="d1",
            chunk_id="c1",
            text="doc1",
            score=0.8,
            parse_hash="ph1",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.8,
            fts_score=0.7,
            vector_rank=1,
            fts_rank=2,
        ),
        SearchResult(
            doc_id="d2",
            chunk_id="c2",
            text="doc2",
            score=0.7,
            parse_hash="ph2",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.7,
            fts_score=0.8,
            vector_rank=2,
            fts_rank=1,
        ),
    ]

    cfg = SearchConfig(embedding_model_id="test-embed")
    monkeypatch.setenv("DASHSCOPE_RERANK_FALLBACK_TO_LANCEDB", "true")
    monkeypatch.setenv("DASHSCOPE_RERANK_RRF_K", "60")

    # Mock _resolve_dashscope_rerank to return our mock
    with patch(
        "xagent.core.tools.core.RAG_tools.pipelines.document_search._resolve_dashscope_rerank",
        return_value=mock_rerank,
    ):
        reranked, used_rerank, warnings = _apply_rerank_if_needed(
            results, "test query", cfg
        )

    # Should fallback to RRF
    assert used_rerank is True
    assert len(reranked) == 2
    assert len(warnings) > 0
    assert any("DashScope rerank failed" in w for w in warnings)
    # RRF should reorder based on ranks (lower rank = better)
    # Both results should be present
    assert all(r.text in ["doc1", "doc2"] for r in reranked)


def test_apply_rerank_rrf_fallback_no_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test RRF fallback fails gracefully when original scores missing."""
    # Create results without original scores/ranks
    results = [
        SearchResult(
            doc_id="d1",
            chunk_id="c1",
            text="doc1",
            score=0.8,
            parse_hash="ph1",
            model_tag="m1",
            created_at=datetime.utcnow(),
        ),
    ]

    cfg = SearchConfig(embedding_model_id="test-embed")
    monkeypatch.setenv("DASHSCOPE_RERANK_FALLBACK_TO_LANCEDB", "true")
    monkeypatch.delenv("DASHSCOPE_RERANK_ENABLED", raising=False)

    # No DashScope rerank available, should try RRF but fail
    reranked, used_rerank, warnings = _apply_rerank_if_needed(
        results, "test query", cfg
    )

    assert used_rerank is False
    assert len(reranked) == 1
    assert len(warnings) > 0
    assert any("missing original" in w.lower() for w in warnings)


def test_apply_rerank_no_rerank_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test rerank skipped when not configured."""
    results = [
        SearchResult(
            doc_id="d1",
            chunk_id="c1",
            text="doc1",
            score=0.8,
            parse_hash="ph1",
            model_tag="m1",
            created_at=datetime.utcnow(),
        ),
    ]

    cfg = SearchConfig(embedding_model_id="test-embed", rerank_model_id=None)
    # Disable rerank and fallback to ensure no rerank is attempted
    monkeypatch.setenv("DASHSCOPE_RERANK_ENABLED", "false")
    monkeypatch.setenv("DASHSCOPE_RERANK_FALLBACK_TO_LANCEDB", "false")

    reranked, used_rerank, warnings = _apply_rerank_if_needed(
        results, "test query", cfg
    )

    assert used_rerank is False
    assert len(reranked) == 1
    assert reranked[0].text == "doc1"
    # No warnings expected when rerank is completely disabled
    assert len(warnings) == 0


def test_apply_rerank_rrf_fallback_with_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test RRF fallback works correctly with original scores/ranks."""
    # Create test results with original scores/ranks
    results = [
        SearchResult(
            doc_id="d1",
            chunk_id="c1",
            text="doc1",
            score=0.8,
            parse_hash="ph1",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.8,
            fts_score=0.7,
            vector_rank=1,
            fts_rank=2,
        ),
        SearchResult(
            doc_id="d2",
            chunk_id="c2",
            text="doc2",
            score=0.7,
            parse_hash="ph2",
            model_tag="m1",
            created_at=datetime.utcnow(),
            vector_score=0.7,
            fts_score=0.8,
            vector_rank=2,
            fts_rank=1,
        ),
    ]

    cfg = SearchConfig(embedding_model_id="test-embed")
    monkeypatch.setenv("DASHSCOPE_RERANK_FALLBACK_TO_LANCEDB", "true")
    monkeypatch.setenv("DASHSCOPE_RERANK_RRF_K", "60")
    monkeypatch.delenv("DASHSCOPE_RERANK_ENABLED", raising=False)

    # No DashScope rerank, should use RRF
    reranked, used_rerank, warnings = _apply_rerank_if_needed(
        results, "test query", cfg
    )

    assert used_rerank is True
    assert len(reranked) == 2
    # RRF should reorder based on ranks
    # Both results should be present, order may vary based on RRF calculation
    assert all(r.text in ["doc1", "doc2"] for r in reranked)
