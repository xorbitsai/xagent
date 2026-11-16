"""Provider-name step of ``LanceDBVectorIndexStore.open_embeddings_table``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DatabaseOperationError
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
    embeddings_table_name,
)

MODEL_ID = "acme-embed"
MODEL_NAME = "acme-embed-v2"
PROVIDER_TABLE = embeddings_table_name(MODEL_NAME)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_embeddings_table,
    )
    from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    lancedb_dir = tmp_path / "open-lancedb"
    lancedb_dir.mkdir()
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
    monkeypatch.delenv("XAGENT_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
    reset_rag_storage_for_tests()
    store = get_vector_index_store()
    ensure_embeddings_table(
        store.get_raw_connection(),
        PROVIDER_TABLE.removeprefix("embeddings_"),
        vector_dim=2,
    )
    assert PROVIDER_TABLE not in embeddings_table_candidates(MODEL_ID)
    yield store
    reset_rag_storage_for_tests()


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    calls: List[str] = []

    def _resolve(model_id: str, **_kwargs: Any) -> Any:
        calls.append(model_id)
        return SimpleNamespace(model_name=MODEL_NAME), object()

    monkeypatch.setattr(model_resolver, "resolve_embedding_adapter", _resolve)
    return calls


def test_the_default_opens_the_provider_name_table(
    store: Any, resolved: List[str]
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )

    table, name = store.open_embeddings_table(MODEL_ID)
    _safe_close_table(table)

    assert name == PROVIDER_TABLE
    assert resolved == [MODEL_ID]


def test_without_the_provider_name_fallback_the_model_is_not_resolved(
    store: Any, resolved: List[str]
) -> None:
    with pytest.raises(DatabaseOperationError):
        store.open_embeddings_table(MODEL_ID, provider_name_fallback=False)

    assert resolved == []
