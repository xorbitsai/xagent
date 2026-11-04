"""Tests for the KB retrieval helper compatibility facade."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Optional

from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult, SearchResult


class _FakeVectorStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.create_index_calls: list[tuple[str, bool]] = []
        self.sync_search_calls: list[dict[str, Any]] = []
        self.async_search_calls: list[dict[str, Any]] = []

    def create_index(self, model_tag: str, readonly: bool) -> IndexResult:
        self.create_index_calls.append((model_tag, readonly))
        return IndexResult(
            status="readonly" if readonly else "index_ready",
            advice="Readonly mode - no index operations" if readonly else None,
            fts_enabled=False,
        )

    def search_vectors_by_model(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.sync_search_calls.append(kwargs)
        return self._matching_rows(
            filters=kwargs["filters"],
            user_id=kwargs["user_id"],
            is_admin=kwargs["is_admin"],
        )

    async def search_vectors_by_model_async(
        self, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.async_search_calls.append(kwargs)
        return self._matching_rows(
            filters=kwargs["filters"],
            user_id=kwargs["user_id"],
            is_admin=kwargs["is_admin"],
        )

    def _matching_rows(
        self,
        filters: Any,
        user_id: Optional[int],
        is_admin: bool,
    ) -> list[dict[str, Any]]:
        rows = [row for row in self._rows if _matches_filter_expr(row, filters)]
        if is_admin:
            return rows
        return [row for row in rows if row.get("user_id") == user_id]


class _FakeStorageShim:
    def __init__(self, vector_store: _FakeVectorStore) -> None:
        self.vector_store = vector_store

    def get_vector_index_store(self) -> _FakeVectorStore:
        return self.vector_store


def _matches_filter_expr(record: dict[str, Any], expr: Any) -> bool:
    if expr is None:
        return True
    if isinstance(expr, (tuple, list)):
        return all(_matches_filter_expr(record, item) for item in expr)

    operator = getattr(expr, "operator", None)
    value = getattr(expr, "value", None)
    field = getattr(expr, "field", "")
    operator_value = getattr(operator, "value", operator)
    record_value = record.get(field)

    if operator_value == "eq":
        return record_value == value
    if operator_value == "gte":
        return record_value >= value
    return False


def _filter_conditions(expr: Any) -> list[tuple[str, str, Any]]:
    if expr is None:
        return []
    if isinstance(expr, (tuple, list)):
        conditions: list[tuple[str, str, Any]] = []
        for item in expr:
            conditions.extend(_filter_conditions(item))
        return conditions
    operator = getattr(expr, "operator", None)
    return [
        (
            getattr(expr, "field"),
            getattr(operator, "value", operator),
            getattr(expr, "value"),
        )
    ]


def _search_row(
    *,
    collection: str = "docs",
    doc_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    user_id: int = 7,
    page_number: int = 3,
    distance: float = 3.0,
) -> dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": f"text for {doc_id}",
        "_distance": distance,
        "parse_hash": f"parse-{doc_id}",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "metadata": '{"page": 3, "section": "intro"}',
        "user_id": user_id,
        "page_number": page_number,
    }


def _signature_shape(callable_obj: Any) -> list[tuple[str, Any, Any]]:
    return [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_obj).parameters.items()
    ]


def test_kb_retrieval_helper_facade_public_surface_imports() -> None:
    """Given the KB package, the retrieval helper facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBRetrievalHelperCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    coordinator = get_kb_coordinator()
    assert isinstance(
        coordinator.retrieval_helper_compatibility,
        KBRetrievalHelperCompatibilityFacade,
    )
    assert coordinator.retrieval_helper is coordinator.retrieval_helper_compatibility


def test_retrieval_facade_preserves_llm_context_formatting() -> None:
    """Given formatting calls through the facade, context strings stay unchanged."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBRetrievalHelperCompatibilityFacade,
    )

    result = SearchResult(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="retrieved text",
        score=0.75,
        parse_hash="parse-1",
        model_tag="model-a",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"page": 1},
    )

    assert KBRetrievalHelperCompatibilityFacade().format_search_results_for_llm(
        [result],
        include_metadata=True,
    ) == (
        "[1]\n"
        "Document ID: doc-1, Chunk ID: chunk-1, Score: 0.7500, "
        "Metadata: {'page': 1}\n"
        "retrieved text"
    )
