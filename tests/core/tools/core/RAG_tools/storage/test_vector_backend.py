"""Tests for vector backend selection."""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.core.tools.core.RAG_tools.storage.vector_backend import (
    VECTOR_BACKEND_ENV,
    VECTOR_BACKEND_ENV_LEGACY,
    VectorBackend,
    get_configured_vector_backend,
)


@pytest.fixture()
def clean_vector_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove backend env vars for isolated parsing."""
    monkeypatch.delenv(VECTOR_BACKEND_ENV, raising=False)
    monkeypatch.delenv(VECTOR_BACKEND_ENV_LEGACY, raising=False)


def test_default_backend_is_lancedb(clean_vector_backend_env: None) -> None:
    assert get_configured_vector_backend() is VectorBackend.LANCEDB


def test_xagent_env_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, clean_vector_backend_env: None
) -> None:
    monkeypatch.setenv(VECTOR_BACKEND_ENV_LEGACY, "milvus")
    monkeypatch.setenv(VECTOR_BACKEND_ENV, "lancedb")
    assert get_configured_vector_backend() is VectorBackend.LANCEDB


def test_legacy_env_when_primary_unset(
    monkeypatch: pytest.MonkeyPatch, clean_vector_backend_env: None
) -> None:
    monkeypatch.setenv(VECTOR_BACKEND_ENV_LEGACY, "lancedb")
    assert get_configured_vector_backend() is VectorBackend.LANCEDB


def test_invalid_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VECTOR_BACKEND_ENV, "not-a-backend")
    with pytest.raises(ConfigurationError, match="Invalid"):
        get_configured_vector_backend()
    # Ensure autouse teardown fixtures can reset storage safely.
    monkeypatch.delenv(VECTOR_BACKEND_ENV, raising=False)


def test_factory_creates_lancedb_store(
    monkeypatch: pytest.MonkeyPatch, clean_vector_backend_env: None, tmp_path: str
) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    monkeypatch.setenv(VECTOR_BACKEND_ENV, "lancedb")
    StorageFactory.get_factory().reset_all()
    store = StorageFactory.get_factory().get_vector_index_store()
    assert isinstance(store, LanceDBVectorIndexStore)


def test_factory_returns_the_lancedb_ledger_for_milvus(
    monkeypatch: pytest.MonkeyPatch, clean_vector_backend_env: None, tmp_path: str
) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    monkeypatch.setenv(VECTOR_BACKEND_ENV, "milvus")
    StorageFactory.get_factory().reset_all()
    store = StorageFactory.get_factory().get_vector_index_store()
    assert isinstance(store, LanceDBVectorIndexStore)
