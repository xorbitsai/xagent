"""Tests for document search pipeline (config coercion and run_document_search)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    SearchConfig,
    SearchPipelineResult,
)
from xagent.core.tools.core.RAG_tools.pipelines import (
    document_search as core_document_search,
)


def test_run_document_search_coerces_mapping(monkeypatch) -> None:
    """Wrapper should convert mapping payloads into ``SearchConfig``."""

    captured: Dict[str, Any] = {}

    def _fake_search_documents(
        collection: str,
        query_text: str,
        *,
        config: SearchConfig,
        progress_manager: Optional[Any] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        **kwargs: Any,
    ) -> SearchPipelineResult:
        captured["collection"] = collection
        captured["query_text"] = query_text
        captured["config"] = config
        return SearchPipelineResult(
            status="success",
            search_type=config.search_type,
            results=[],
            result_count=0,
            warnings=[],
            message="ok",
            used_rerank=False,
        )

    monkeypatch.setattr(
        core_document_search, "search_documents", _fake_search_documents
    )

    result = core_document_search.run_document_search(
        "demo", "hello", config={"embedding_model_id": "fake-model"}
    )

    assert isinstance(result, SearchPipelineResult)
    assert captured["collection"] == "demo"
    assert captured["query_text"] == "hello"
    assert isinstance(captured["config"], SearchConfig)
    assert captured["config"].embedding_model_id == "fake-model"


def test_run_document_search_rejects_invalid_config() -> None:
    """Wrapper should fail fast when search_config is not coercible."""

    with pytest.raises(TypeError):
        core_document_search.run_document_search(
            "demo",
            "hello",
            config=42,  # type: ignore[arg-type]
        )
