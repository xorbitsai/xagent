"""Tests for model resolver utilities."""

from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
import time
from typing import Dict

import pytest
from sqlalchemy.exc import OperationalError as SAOperationalError

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    RerankModelConfig,
)
from xagent.core.model.rerank.base import BaseRerank
from xagent.core.model.storage.error import ModelNotFoundError
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    EmbeddingAdapterError,
    RagCoreException,
)
from xagent.core.tools.core.RAG_tools.utils import model_resolver


class _StubHub:
    """Stub hub for testing."""

    def __init__(self, models: Dict[str, object]) -> None:
        self._models = models

    def list(self) -> Dict[str, object]:
        return self._models

    def load(self, model_id: str, **_kwargs: object) -> object:
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]


class TestHubInitFailureClassification:
    """Tests for _hub_init_failure_is_benign_optional_sqlite."""

    def test_sqlite_missing_file_is_benign(self) -> None:
        exc = sqlite3.OperationalError("unable to open database file")
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is True

    def test_sqlalchemy_wrapped_sqlite_missing_is_benign(self) -> None:
        inner = sqlite3.OperationalError("unable to open database file")
        exc = SAOperationalError("SELECT 1", {}, inner)
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is True

    def test_database_locked_not_benign(self) -> None:
        exc = sqlite3.OperationalError("database is locked")
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is False

    def test_other_errors_not_benign(self) -> None:
        assert (
            model_resolver._hub_init_failure_is_benign_optional_sqlite(
                RuntimeError("connection refused")
            )
            is False
        )


class TestGetOrInitModelHub:
    """Test _get_or_init_model_hub helper function."""

    def test_get_existing_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting existing hub."""
        stub_hub = _StubHub({})
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)
        result = model_resolver._get_or_init_model_hub()
        assert result == stub_hub

    def test_get_or_init_model_hub_initializes_cache_once_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent callers should share one initialized engine/sessionmaker."""
        model_resolver._reset_model_hub_cache()

        worker_count = 5
        barrier = threading.Barrier(worker_count)
        guard = threading.Lock()
        create_calls: list[object] = []
        sessionmaker_calls: list[object] = []
        sessions: list[object] = []

        class FakeEngine:
            def __init__(self) -> None:
                self.disposed = False

            def dispose(self) -> None:
                self.disposed = True

        class FakeMetadata:
            def create_all(self, engine: object) -> None:
                time.sleep(0.02)

        class FakeBase:
            metadata = FakeMetadata()

        class FakeSessionLocal:
            def __call__(self) -> object:
                session = object()
                with guard:
                    sessions.append(session)
                return session

        class FakeHub:
            def __init__(self, db: object, model: object) -> None:
                self.db = db
                self.model = model

        fake_session_local = FakeSessionLocal()
        fake_model = object()

        def fake_create_engine(*args: object, **kwargs: object) -> FakeEngine:
            engine = FakeEngine()
            with guard:
                create_calls.append(engine)
            time.sleep(0.05)
            return engine

        def fake_sessionmaker(*args: object, **kwargs: object) -> FakeSessionLocal:
            with guard:
                sessionmaker_calls.append(kwargs.get("bind"))
            return fake_session_local

        def worker() -> FakeHub:
            barrier.wait(timeout=5)
            hub = model_resolver._get_or_init_model_hub()
            assert isinstance(hub, FakeHub)
            return hub

        monkeypatch.setattr(
            model_resolver, "get_default_db_url", lambda: "sqlite:///concurrent.db"
        )
        monkeypatch.setattr(model_resolver, "create_engine", fake_create_engine)
        monkeypatch.setattr(model_resolver, "declarative_base", lambda: FakeBase)
        monkeypatch.setattr(
            model_resolver, "create_model_table", lambda Base: fake_model
        )
        monkeypatch.setattr(model_resolver, "sessionmaker", fake_sessionmaker)
        monkeypatch.setattr(model_resolver, "SQLAlchemyModelHub", FakeHub)

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as executor:
                hubs = list(executor.map(lambda _: worker(), range(worker_count)))

            assert len(create_calls) == 1
            assert len(sessionmaker_calls) == 1
            assert len(sessions) == worker_count
            assert all(hub.model is fake_model for hub in hubs)
        finally:
            model_resolver._reset_model_hub_cache()

    def test_reset_model_hub_cache_disposes_engine(self) -> None:
        """Reset should dispose the active engine and clear the resource group."""

        class FakeEngine:
            def __init__(self) -> None:
                self.disposed = False

            def dispose(self) -> None:
                self.disposed = True

        engine = FakeEngine()
        model_resolver._MODEL_HUB_ENGINE = engine
        model_resolver._MODEL_HUB_SESSION_LOCAL = object()
        model_resolver._MODEL_HUB_MODEL = object()
        model_resolver._MODEL_HUB_DB_URL = "sqlite:///old.db"

        model_resolver._reset_model_hub_cache()

        assert engine.disposed is True
        assert model_resolver._MODEL_HUB_ENGINE is None
        assert model_resolver._MODEL_HUB_SESSION_LOCAL is None
        assert model_resolver._MODEL_HUB_MODEL is None
        assert model_resolver._MODEL_HUB_DB_URL is None

    def test_get_or_init_model_hub_re_raises_non_db_reinit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-DB init failures should not be hidden as hub unavailability."""

        class FakeEngine:
            def __init__(self) -> None:
                self.disposed = False

            def dispose(self) -> None:
                self.disposed = True

        class FakeMetadata:
            def create_all(self, engine: object) -> None:
                raise RuntimeError("create failed")

        class FakeBase:
            metadata = FakeMetadata()

        old_engine = FakeEngine()
        new_engine = FakeEngine()
        old_session_local = object()
        old_model = object()
        model_resolver._MODEL_HUB_ENGINE = old_engine
        model_resolver._MODEL_HUB_SESSION_LOCAL = old_session_local
        model_resolver._MODEL_HUB_MODEL = old_model
        model_resolver._MODEL_HUB_DB_URL = "sqlite:///old.db"

        monkeypatch.setattr(
            model_resolver, "get_default_db_url", lambda: "sqlite:///new.db"
        )
        monkeypatch.setattr(
            model_resolver, "create_engine", lambda *args, **kwargs: new_engine
        )
        monkeypatch.setattr(model_resolver, "declarative_base", lambda: FakeBase)
        monkeypatch.setattr(model_resolver, "create_model_table", lambda Base: object())

        try:
            with pytest.raises(RuntimeError, match="create failed"):
                model_resolver._get_or_init_model_hub()

            assert new_engine.disposed is True
            assert old_engine.disposed is False
            assert model_resolver._MODEL_HUB_ENGINE is old_engine
            assert model_resolver._MODEL_HUB_SESSION_LOCAL is old_session_local
            assert model_resolver._MODEL_HUB_MODEL is old_model
            assert model_resolver._MODEL_HUB_DB_URL == "sqlite:///old.db"
        finally:
            model_resolver._reset_model_hub_cache()

    def test_get_or_init_model_hub_returns_none_for_recoverable_db_reinit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recoverable DB init failures may fall back without polluting cache."""

        class FakeEngine:
            def __init__(self) -> None:
                self.disposed = False

            def dispose(self) -> None:
                self.disposed = True

        class FakeMetadata:
            def create_all(self, engine: object) -> None:
                raise SAOperationalError(
                    "SELECT models",
                    {},
                    Exception("too many clients already"),
                )

        class FakeBase:
            metadata = FakeMetadata()

        old_engine = FakeEngine()
        new_engine = FakeEngine()
        old_session_local = object()
        old_model = object()
        model_resolver._MODEL_HUB_ENGINE = old_engine
        model_resolver._MODEL_HUB_SESSION_LOCAL = old_session_local
        model_resolver._MODEL_HUB_MODEL = old_model
        model_resolver._MODEL_HUB_DB_URL = "sqlite:///old.db"

        monkeypatch.setattr(
            model_resolver, "get_default_db_url", lambda: "sqlite:///new.db"
        )
        monkeypatch.setattr(
            model_resolver, "create_engine", lambda *args, **kwargs: new_engine
        )
        monkeypatch.setattr(model_resolver, "declarative_base", lambda: FakeBase)
        monkeypatch.setattr(model_resolver, "create_model_table", lambda Base: object())

        try:
            assert model_resolver._get_or_init_model_hub() is None

            assert new_engine.disposed is True
            assert old_engine.disposed is False
            assert model_resolver._MODEL_HUB_ENGINE is old_engine
            assert model_resolver._MODEL_HUB_SESSION_LOCAL is old_session_local
            assert model_resolver._MODEL_HUB_MODEL is old_model
            assert model_resolver._MODEL_HUB_DB_URL == "sqlite:///old.db"
        finally:
            model_resolver._reset_model_hub_cache()


class TestResolveEmbeddingAdapter:
    """Test resolve_embedding_adapter function with strict priority."""

    def test_resolve_embedding_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-model": EmbeddingModelConfig(
                    id="hub-model",
                    model_name="hub-model",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["embedding"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Set env vars (should be ignored when model_id is explicit)
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id="hub-model")
        assert cfg.id == "hub-model"
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": EmbeddingModelConfig(
                    id="default",
                    model_name="hub-embedding",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["embedding"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_EMBEDDING_MODEL",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_EMBEDDING_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id=None)
        assert cfg.id == "default"
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding from env when hub fails (fallback)."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Set env vars for fallback
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        monkeypatch.setenv(
            "DASHSCOPE_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        )
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_DIMENSION", "1536")

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id=None)
        assert cfg.id == "env-model"
        assert cfg.dimension == 1536
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_does_not_fallback_for_non_db_init_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-DB hub initialization errors should surface even when env exists."""

        def buggy_hub() -> object:
            raise RuntimeError("broken model hub initialization")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", buggy_hub)
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

        with pytest.raises(RuntimeError, match="broken model hub initialization"):
            model_resolver.resolve_embedding_adapter(model_id=None)

    def test_resolve_embedding_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Clear env vars
        for key in [
            "DASHSCOPE_EMBEDDING_MODEL",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_EMBEDDING_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(EmbeddingAdapterError):
            model_resolver.resolve_embedding_adapter(model_id=None)


class TestResolveRerankAdapter:
    """Test resolve_rerank_adapter function with strict priority."""

    def test_resolve_rerank_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving rerank with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-rerank": RerankModelConfig(
                    id="hub-rerank",
                    model_name="hub-rerank",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["rerank"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Set env vars (should be ignored when model_id is explicit)
        monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "env-rerank")
        monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "env-key")

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id="hub-rerank")
        assert cfg.id == "hub-rerank"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving rerank for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": RerankModelConfig(
                    id="default",
                    model_name="hub-rerank",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["rerank"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_RERANK_MODEL",
            "DASHSCOPE_RERANK_API_KEY",
            "DASHSCOPE_RERANK_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id=None)
        assert cfg.id == "default"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving rerank from env when hub fails (fallback)."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Set env vars for fallback
        monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "env-rerank")
        monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "env-key")
        monkeypatch.setenv(
            "DASHSCOPE_RERANK_BASE_URL",
            "https://dashscope.aliyuncs.com/rerank",
        )
        monkeypatch.setenv("DASHSCOPE_RERANK_TIMEOUT", "30")

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id=None)
        assert cfg.id == "env-rerank"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Clear env vars
        for key in [
            "DASHSCOPE_RERANK_MODEL",
            "DASHSCOPE_RERANK_API_KEY",
            "DASHSCOPE_RERANK_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(RagCoreException):
            model_resolver.resolve_rerank_adapter(model_id=None)


class TestResolveLLMAdapter:
    """Test resolve_llm_adapter function."""

    def test_resolve_llm_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving LLM with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-llm": ChatModelConfig(
                    id="hub-llm",
                    model_name="hub-llm",
                    model_provider="openai",
                    api_key="hub-key",
                    abilities=["chat"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id="hub-llm", use_langchain_adapter=False
        )
        assert cfg.id == "hub-llm"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving LLM for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": ChatModelConfig(
                    id="default",
                    model_name="hub-llm",
                    model_provider="openai",
                    api_key="hub-key",
                    abilities=["chat"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars to ensure hub is used
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_NAME",
            "ZHIPU_API_KEY",
            "DEEPSEEK_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id=None, use_langchain_adapter=False
        )
        assert cfg.id == "default"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving LLM from env when hub fails (fallback)."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Set env vars for fallback (OpenAI)
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id=None, use_langchain_adapter=False
        )
        assert cfg.id == "gpt-4"
        assert cfg.model_provider == "openai"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""
        # Mock recoverable hub unavailability.
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: None)

        # Clear env vars
        for key in ["OPENAI_API_KEY", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(RagCoreException):
            model_resolver.resolve_llm_adapter(
                model_id=None, use_langchain_adapter=False
            )
