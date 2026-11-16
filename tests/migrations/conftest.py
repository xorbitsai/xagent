"""Fixtures of the embedding migration tests; none is autouse."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from tests.migrations.embedding_migration_shared import (
    SOURCE,
    TARGET,
    _Adapter,
    _config,
    _create_table,
    _Hub,
)
from xagent.migrations.embedding import scope


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_Hub]:
    """A temporary LanceDB directory and SQL file; the hub is served in memory."""
    from xagent.core.tools.core.RAG_tools.storage import reset_rag_storage_for_tests
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    lancedb_dir = tmp_path / "warmup-lancedb"
    lancedb_dir.mkdir()
    monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'hub.db'}")
    model_resolver._reset_model_hub_cache()
    served = _Hub({SOURCE: _config(SOURCE), TARGET: _config(TARGET)})
    monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: served)
    reset_rag_storage_for_tests()
    _create_table(scope.source_table(TARGET))
    yield served
    reset_rag_storage_for_tests()
    model_resolver._reset_model_hub_cache()


@pytest.fixture
def index_calls(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBVectorIndexStore,
    )

    record = SimpleNamespace(calls=[], status="below_threshold")

    def _create_index(self: Any, model_tag: str, readonly: bool = False) -> Any:
        record.calls.append((model_tag, readonly))
        return IndexResult(status=record.status, advice=None, fts_enabled=True)

    monkeypatch.setattr(LanceDBVectorIndexStore, "create_index", _create_index)
    return record


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    built = _Adapter()

    def _factory(*args: Any, **kwargs: Any) -> _Adapter:
        built.builds += 1
        return built

    monkeypatch.setattr(model_resolver, "create_embedding_adapter", _factory)
    return built
