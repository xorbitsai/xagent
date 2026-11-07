"""Contract tests for KB/RAG user scope context propagation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import SearchConfig, SearchType
from xagent.core.tools.core.RAG_tools.pipelines import (
    document_ingestion,
    document_search,
)
from xagent.core.tools.core.RAG_tools.utils.user_scope import (
    get_user_scope,
    resolve_user_scope,
    user_scope_context,
)


def test_resolve_user_scope_prefers_explicit_values() -> None:
    """Explicit scope arguments should override context values."""
    with user_scope_context(user_id=10, is_admin=False):
        resolved = resolve_user_scope(user_id=99, is_admin=True)
    assert resolved.user_id == 99
    assert resolved.is_admin is True


def test_resolve_user_scope_none_falls_back_to_context() -> None:
    """When user_id and is_admin are both None, should fall back to context."""
    with user_scope_context(user_id=42, is_admin=True):
        resolved = resolve_user_scope(user_id=None, is_admin=None)
    assert resolved.user_id == 42
    assert resolved.is_admin is True


def test_resolve_user_scope_explicit_false_is_admin() -> None:
    """Explicit is_admin=False should not fall back to context even if context has True."""
    with user_scope_context(user_id=10, is_admin=True):
        resolved = resolve_user_scope(user_id=None, is_admin=False)
    assert resolved.user_id is None
    assert resolved.is_admin is False


def test_user_scope_context_resets_after_exit() -> None:
    """Context should be restored after exiting scope manager."""
    before = get_user_scope()
    with user_scope_context(user_id=7, is_admin=False):
        during = get_user_scope()
        assert during.user_id == 7
    after = get_user_scope()
    assert after == before


def test_context_isolation_between_async_tasks() -> None:
    """Async tasks should keep their own context values."""

    async def _worker(user_id: int) -> int:
        with user_scope_context(user_id=user_id, is_admin=False):
            await asyncio.sleep(0)
            scope = get_user_scope()
            assert scope.user_id == user_id
            return int(scope.user_id or -1)

    async def _main() -> list[int]:
        return list(await asyncio.gather(_worker(1), _worker(2)))

    result = asyncio.run(_main())
    assert sorted(result) == [1, 2]


def test_run_document_ingestion_omitted_scope_falls_back_to_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document ingestion entrypoint should use request scope when args are omitted."""
    captured: dict[str, Any] = {}
    expected_result = object()

    def _fake_process_document(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(document_ingestion, "process_document", _fake_process_document)

    with user_scope_context(user_id=42, is_admin=True):
        result = document_ingestion.run_document_ingestion(
            collection="ctx_collection",
            source_path="/tmp/source.md",
        )

    assert result is expected_result
    assert captured["user_id"] == 42
    assert captured["is_admin"] is True


def test_run_document_search_omitted_scope_preserves_context_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document search wrapper should not pass explicit False when scope is omitted."""
    captured: dict[str, Any] = {}

    def _fake_search_documents(
        collection: str,
        query_text: str,
        **kwargs: Any,
    ) -> object:
        scope = resolve_user_scope(
            user_id=kwargs.get("user_id"),
            is_admin=kwargs.get("is_admin"),
        )
        captured["user_id"] = scope.user_id
        captured["is_admin"] = scope.is_admin
        return object()

    monkeypatch.setattr(document_search, "search_documents", _fake_search_documents)

    with user_scope_context(user_id=43, is_admin=True):
        document_search.run_document_search(
            collection="ctx_collection",
            query_text="query",
            config={"embedding_model_id": "fake-model"},
        )

    assert captured["user_id"] == 43
    assert captured["is_admin"] is True


def _patch_search_pipeline_common(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int | None, bool]]:
    """Patch expensive search dependencies and capture sparse search scope."""
    sparse_calls: list[tuple[int | None, bool]] = []

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager."
        "resolve_effective_embedding_model_sync",
        lambda _collection, model_id: model_id,
    )
    monkeypatch.setattr(
        document_search,
        "resolve_embedding_adapter",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        document_search,
        "_apply_rerank_if_needed",
        lambda results, _query_text, _cfg: (results, False, []),
    )

    def _fake_sparse_search(
        collection: str,
        query_text: str,
        cfg: SearchConfig,
        model_tag: str,
        user_id: int | None = None,
        is_admin: bool = False,
    ) -> tuple[list[Any], str, list[str], str]:
        sparse_calls.append((user_id, is_admin))
        return [], "success", [], "ok"

    monkeypatch.setattr(document_search, "_execute_sparse_search", _fake_sparse_search)
    return sparse_calls


def test_search_documents_omitted_scope_falls_back_to_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search entrypoint should use request scope when args are omitted."""
    sparse_calls = _patch_search_pipeline_common(monkeypatch)

    with user_scope_context(user_id=44, is_admin=True):
        result = document_search.search_documents(
            collection="ctx_collection",
            query_text="query",
            config=SearchConfig(
                search_type=SearchType.SPARSE,
                embedding_model_id="fake-model",
            ),
        )

    assert result.status == "success"
    assert sparse_calls == [(44, True)]


def test_search_documents_embedding_fallback_preserves_resolved_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid embedding failure fallback should keep the resolved user scope."""
    sparse_calls = _patch_search_pipeline_common(monkeypatch)

    def _raise_vector_validation_error(*args: Any, **kwargs: Any) -> list[float]:
        raise VectorValidationError("bad vector")

    monkeypatch.setattr(
        document_search,
        "_encode_query_vector",
        _raise_vector_validation_error,
    )

    result = document_search.search_documents(
        collection="ctx_collection",
        query_text="query",
        config=SearchConfig(
            search_type=SearchType.HYBRID,
            embedding_model_id="fake-model",
            fallback_to_sparse=True,
        ),
        user_id=45,
        is_admin=True,
    )

    assert result.status == "success"
    assert sparse_calls == [(45, True)]


def test_search_documents_hybrid_failure_fallback_preserves_resolved_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid search failure fallback should keep the resolved user scope."""
    sparse_calls = _patch_search_pipeline_common(monkeypatch)

    monkeypatch.setattr(document_search, "_encode_query_vector", lambda *_args: [0.1])

    def _raise_hybrid_error(**kwargs: Any) -> object:
        raise ValueError("hybrid failed")

    monkeypatch.setattr(document_search, "search_hybrid", _raise_hybrid_error)

    result = document_search.search_documents(
        collection="ctx_collection",
        query_text="query",
        config=SearchConfig(
            search_type=SearchType.HYBRID,
            embedding_model_id="fake-model",
            fallback_to_sparse=True,
        ),
        user_id=46,
        is_admin=True,
    )

    assert result.status == "success"
    assert sparse_calls == [(46, True)]


def test_ingestion_resolves_its_embedding_model_in_the_callers_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A bare model name must resolve among the ingesting user's models."""
    seen: list[int | None] = []

    def _capture_scope(_cfg: Any) -> Any:
        seen.append(get_user_scope().user_id)
        raise RuntimeError("stop after resolving the embedding model")

    monkeypatch.setattr(
        document_ingestion, "validate_document_processing_sync", lambda **_kw: None
    )
    monkeypatch.setattr(
        document_ingestion,
        "resolve_effective_embedding_model_sync",
        lambda _collection, model_id: model_id,
    )
    monkeypatch.setattr(
        document_ingestion, "_resolve_embedding_adapter", _capture_scope
    )
    source = tmp_path / "doc.md"
    source.write_text("hello")

    result = document_ingestion._process_document_impl(
        "ctx_collection", str(source), user_id=48, is_admin=False
    )

    assert result.failed_step == "resolve_embedding_adapter"
    assert seen == [48]
    assert get_user_scope().user_id is None


def test_search_resolves_its_embedding_model_in_the_callers_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit search scope must reach the embedding and rerank resolvers."""
    _patch_search_pipeline_common(monkeypatch)
    seen: list[int | None] = []

    def _capture_scope(*_args: Any, **_kwargs: Any) -> tuple[object, object]:
        seen.append(get_user_scope().user_id)
        return object(), object()

    def _capture_rerank_scope(
        results: Any, _query_text: str, _cfg: SearchConfig
    ) -> tuple[Any, bool, list[str]]:
        seen.append(get_user_scope().user_id)
        return results, False, []

    monkeypatch.setattr(document_search, "resolve_embedding_adapter", _capture_scope)
    monkeypatch.setattr(
        document_search, "_apply_rerank_if_needed", _capture_rerank_scope
    )

    document_search.search_documents(
        collection="ctx_collection",
        query_text="query",
        config=SearchConfig(
            search_type=SearchType.SPARSE,
            embedding_model_id="fake-model",
        ),
        user_id=49,
        is_admin=False,
    )

    assert seen == [49, 49]
