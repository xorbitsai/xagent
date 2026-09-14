"""
Shared fixtures for web integration tests.

This module provides common fixtures used across all E2E web integration tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.auth_database import auth_db_override
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User

# ==========================================
# SHARED STUBS FOR CONTRACT_STUB TESTS
# ==========================================


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for E2E contract tests.

    This stub provides predictable, deterministic embeddings for testing
    without requiring real embedding model API calls. The embedding
    is simply [len(text), 0.0] for any input text.
    """

    def encode(
        self,
        text: Any,
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Any:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> list[str]:
        return ["embedding"]


@pytest.fixture
def stub_embedding_config() -> EmbeddingModelConfig:
    """Create stub embedding configuration for E2E tests.

    This is a default fixture that can be overridden by individual test files.
    Test files that need specific embedding_model_id values should define
    their own stub_embedding_config fixture.
    """
    return EmbeddingModelConfig(
        id="e2e-test-embedding",
        model_name="e2e-test-embedding-model",
        model_provider="test",
        dimension=2,
    )


# Provide a non-autouse version that test files can explicitly depend on
# Helper function to set up RAG mocks (used by test-specific fixtures)
def _setup_rag_mocks(
    monkeypatch: Any,
    stub_embedding_config: EmbeddingModelConfig,
    stub_embedding_adapter: _StubEmbeddingAdapter,
) -> None:
    """Set up RAG pipeline mocks for E2E testing."""
    from xagent.core.tools.core.RAG_tools import pipelines as pipelines_module
    from xagent.core.tools.core.RAG_tools.management import collection_manager
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    mgr = collection_manager.collection_manager

    async def mock_get_collection(collection_name: str) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name,
            embedding_model_id=stub_embedding_config.id,
            embedding_dimension=2,
        )

    async def mock_initialize_collection(
        collection_name: str, embedding_model_id: str
    ) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_dimension=2,
        )

    def mock_resolve_embedding_adapter(
        model_id: str | None = None, **kwargs: Any
    ) -> tuple[EmbeddingModelConfig, BaseEmbedding]:
        return (stub_embedding_config, stub_embedding_adapter)

    monkeypatch.setattr(mgr, "get_collection", mock_get_collection)
    monkeypatch.setattr(
        mgr, "initialize_collection_embedding", mock_initialize_collection
    )
    monkeypatch.setattr(
        model_resolver, "resolve_embedding_adapter", mock_resolve_embedding_adapter
    )
    monkeypatch.setattr(
        pipelines_module.document_ingestion,
        "_resolve_embedding_adapter",
        lambda cfg: (stub_embedding_config, stub_embedding_adapter),
    )


@pytest.fixture
def stub_embedding_adapter() -> _StubEmbeddingAdapter:
    """Create stub embedding adapter for E2E tests."""
    return _StubEmbeddingAdapter()


@pytest.fixture(autouse=True)
def mock_rag_pipeline(
    monkeypatch: Any,
    request: pytest.FixtureRequest,
    stub_embedding_adapter: _StubEmbeddingAdapter,
) -> None:
    """Mock the RAG pipeline components for E2E testing.

    This fixture uses the test file's own stub_embedding_config if defined,
    otherwise falls back to the default from this conftest.py.
    """
    # Do not monkeypatch embedding pipeline for real provider smoke tests.
    # Those tests must exercise the real embedding/search stack.
    if request.node.get_closest_marker("real_rag") is not None:
        return

    # Get stub_embedding_config from the test file if available, otherwise use default
    config = request.getfixturevalue("stub_embedding_config")
    _setup_rag_mocks(monkeypatch, config, stub_embedding_adapter)


@pytest.fixture(scope="function")
def test_env(monkeypatch: pytest.MonkeyPatch):
    """Setup test database and app for E2E tests."""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    TestingSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Web ingestion runs file persistence inside run_in_executor; it opens sessions via
    # get_session_local(), which reads module globals—not dependency_overrides[get_db].
    # Align executor-thread sessions with the same test factory used by FastAPI Depends.
    monkeypatch.setattr(
        "xagent.web.api.kb.get_session_local",
        lambda: TestingSessionLocal,
    )

    app = FastAPI()
    app.include_router(kb_router)
    # Include auth router for register/login endpoints
    from xagent.web.api.auth import auth_router

    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_auth_db] = auth_db_override(override_get_db)

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    user = User(
        username="testuser", password_hash=hash_password("test"), is_admin=False
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Mock JWT token (must include type="access" for get_current_user)
    from datetime import datetime, timedelta, timezone

    import jwt

    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}

    yield app, headers, user, TestingSessionLocal

    session.close()
    test_engine.dispose()  # MAJOR #10: Dispose engine before unlinking DB
    os.unlink(temp_db_path)


@pytest.fixture
def client(test_env):
    """Provide test client for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return TestClient(app)


@pytest.fixture
def auth_headers(test_env):
    """Provide authentication headers for E2E tests."""
    app, headers, user, TestingSessionLocal = test_env
    return headers


@pytest.fixture
def db_session_factory(test_env):
    """Expose SQLAlchemy session factory for test data adjustments."""
    app, headers, user, TestingSessionLocal = test_env
    return TestingSessionLocal


@pytest.fixture
def _legacy_clean_storage_compat() -> None:
    """Legacy no-op fixture for backward compatibility (MINOR #14).

    This fixture exists only for backward compatibility with older tests.
    Actual storage cleanup is handled by the root ``isolate_rag_storage``
    autouse fixture in ``tests/conftest.py``.

    Tests should NOT use this fixture directly - it will be removed in a future update.
    """
    return None


@pytest.fixture(autouse=True)
def temp_uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup temporary uploads directory and patch upload path functions (MAJOR #4).

    This ensures all file uploads during tests go to a temporary directory
    instead of the real upload directory, providing proper test isolation.
    """
    from unittest.mock import patch

    def patched_get_upload_path(
        filename,
        task_id=None,
        folder=None,
        user_id=None,
        collection=None,
        create_if_not_exists=True,
        collection_is_sanitized=False,
    ):
        """Patched version that uses temp_path instead of real upload path."""
        base = tmp_path
        if user_id:
            user_dir = base / f"user_{user_id}"
            if collection:
                d = user_dir / collection
                if create_if_not_exists:
                    d.mkdir(parents=True, exist_ok=True)
                return d / filename
            if create_if_not_exists:
                user_dir.mkdir(parents=True, exist_ok=True)
            return user_dir / filename
        return base / filename

    with (
        patch(
            "xagent.web.api.kb.get_upload_path",
            side_effect=patched_get_upload_path,
        ),
        patch(
            "xagent.web.services.kb_collection_service.get_upload_path",
            side_effect=patched_get_upload_path,
        ),
        patch(
            "xagent.web.config.get_upload_path",
            side_effect=patched_get_upload_path,
        ),
        patch("xagent.config.get_uploads_dir", return_value=tmp_path),
        patch(
            "xagent.web.services.kb_file_service.get_uploads_dir",
            return_value=tmp_path,
        ),
        patch(
            "xagent.web.services.kb_collection_service.get_uploads_dir",
            return_value=tmp_path,
        ),
    ):
        yield tmp_path
