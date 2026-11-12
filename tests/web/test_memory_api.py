"""Test cases for memory API endpoints."""

import json
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from unittest.mock import Mock
from urllib.parse import quote

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote, MemoryResponse
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.storage_admission import (
    DormantLanceDBMemoryHandle,
    StorageAdmissionState,
    admit_lancedb_memory_storage,
)
from xagent.core.memory.vector_compatibility import EmbeddingIdentity
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.web.api.auth import auth_router, hash_password
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User
from xagent.web.services.memory_binding import (
    MemoryManagementBindingRequest,
    MemoryStoreBinding,
    set_trusted_memory_binding_resolver,
)
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

# Fixture IDs in the minted str(uuid.uuid4()) form; the routes accept any opaque ID.
MEMORY_ID_1 = "0b7a6a52-3f1e-4c3a-9d55-1f0c8e2a7b11"
MEMORY_ID_2 = "5d2c9e40-8a6b-4f7d-b3c1-6e9f2a4d8c22"
MISSING_MEMORY_ID = "9e8d7c6b-5a49-4382-a716-2f3e4d5c6b33"
OTHER_MEMORY_ID = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c44"
CRD_MEMORY_ID = "7f6e5d4c-3b2a-4190-8f7e-6d5c4b3a2955"

# Create temporary directory for database
temp_dir = tempfile.mkdtemp()
temp_db_path = os.path.join(temp_dir, "test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def clear_memory_binding_resolver() -> Iterator[None]:
    set_trusted_memory_binding_resolver(None)
    yield
    set_trusted_memory_binding_resolver(None)


def override_get_db():
    db = None
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    # Create admin user
    session = TestingSessionLocal()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        session.add(admin_user)
        session.commit()
        yield admin_user
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def cleanup_global_test_db():
    """Cleanup global test database after all tests"""
    yield
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="function")
def auth_headers(test_db):
    """Authentication headers for admin user"""
    # Create a valid JWT token directly
    from datetime import datetime, timedelta, timezone

    import jwt

    payload = {
        "sub": "admin",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "user_id": test_db.id,  # Use actual user ID from test_db fixture
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    store = Mock(spec=MemoryStore)
    return store


@pytest.fixture
def memory_router(mock_memory_store):
    """Create memory management router with mock store."""
    return MemoryManagementRouter(lambda: mock_memory_store)


@pytest.fixture
def client(memory_router):
    """Create test client with memory router and authentication."""
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(memory_router.get_router())
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
def sample_memory_notes():
    """Create sample memory notes for testing."""
    return [
        MemoryNote(
            id=MEMORY_ID_1,
            content="Test memory 1",
            keywords=["test", "sample"],
            tags=["important"],
            category="general",
            metadata={"source": "test"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
        MemoryNote(
            id=MEMORY_ID_2,
            content="Test memory 2",
            keywords=["test", "example"],
            tags=["normal"],
            category="system",
            metadata={"source": "system"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
    ]


def test_management_binding_resolves_once_per_request_after_authentication(
    auth_headers: dict[str, str],
    test_db: User,
) -> None:
    selected_store = InMemoryMemoryStore()
    legacy_provider = Mock(side_effect=AssertionError("legacy store must not be used"))
    requests: list[MemoryManagementBindingRequest] = []

    def resolver(request: object) -> MemoryStoreBinding:
        assert isinstance(request, MemoryManagementBindingRequest)
        requests.append(request)
        return MemoryStoreBinding(
            store=selected_store,
            backend_identity="backend-a",
            route_generation="route-1",
            public_diagnostics={"backend": "host-managed", "ready": True},
        )

    set_trusted_memory_binding_resolver(resolver)
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(MemoryManagementRouter(legacy_provider).get_router())
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    created = client.post(
        "/api/memory",
        json={"content": "bound note"},
        headers=auth_headers,
    )
    info = client.get("/api/memory/store-info", headers=auth_headers)

    assert created.status_code == 200
    assert info.status_code == 200
    assert info.json() == {"backend": "host-managed", "ready": True}
    assert requests == [
        MemoryManagementBindingRequest(authenticated_user_id=int(test_db.id)),
        MemoryManagementBindingRequest(authenticated_user_id=int(test_db.id)),
    ]
    legacy_provider.assert_not_called()
    assert selected_store.get(created.json()["memory_id"]).success


def test_management_binding_failure_returns_503_without_legacy_fallback(
    auth_headers: dict[str, str],
) -> None:
    legacy_provider = Mock(side_effect=AssertionError("legacy store must not be used"))
    set_trusted_memory_binding_resolver(
        Mock(side_effect=RuntimeError("backend unavailable"))
    )
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(MemoryManagementRouter(legacy_provider).get_router())
    app.dependency_overrides[get_db] = override_get_db

    response = TestClient(app).get("/api/memory/list", headers=auth_headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "Memory storage is temporarily unavailable."}
    legacy_provider.assert_not_called()


def test_management_binding_is_not_resolved_before_authentication() -> None:
    resolver = Mock(side_effect=AssertionError("resolver must run after auth"))
    set_trusted_memory_binding_resolver(resolver)
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(MemoryManagementRouter().get_router())
    app.dependency_overrides[get_db] = override_get_db

    response = TestClient(app).get("/api/memory/list")

    assert response.status_code == 401
    resolver.assert_not_called()


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        ("post", "/api/memory", {"json": {}}),
        ("post", "/api/memory", {"json": {"content": " \t\n"}}),
        (
            "post",
            "/api/memory",
            {"content": b"", "headers": {"content-type": "application/json"}},
        ),
        ("put", f"/api/memory/{MEMORY_ID_1}", {"json": {"content": ""}}),
        (
            "put",
            f"/api/memory/{MEMORY_ID_1}",
            {"content": b"", "headers": {"content-type": "application/json"}},
        ),
    ],
)
def test_invalid_body_is_rejected_before_remote_binding_resolution(
    client: TestClient,
    mock_memory_store: Mock,
    auth_headers: dict[str, str],
    method: str,
    path: str,
    request_kwargs: dict[str, Any],
) -> None:
    resolver = Mock(side_effect=AssertionError("invalid body must not bind"))
    set_trusted_memory_binding_resolver(resolver)
    kwargs = dict(request_kwargs)
    headers = {**auth_headers, **kwargs.pop("headers", {})}

    response = client.request(method, path, headers=headers, **kwargs)

    assert response.status_code == 422
    resolver.assert_not_called()
    mock_memory_store.add.assert_not_called()
    mock_memory_store.get.assert_not_called()
    mock_memory_store.update.assert_not_called()


def test_update_empty_object_keeps_existing_noop_contract(
    client: TestClient,
    mock_memory_store: Mock,
    sample_memory_notes: list[MemoryNote],
    auth_headers: dict[str, str],
) -> None:
    mock_memory_store.get.return_value = Mock(
        success=True, content=sample_memory_notes[0]
    )
    mock_memory_store.update.return_value = Mock(success=True)

    response = client.put(f"/api/memory/{MEMORY_ID_1}", json={}, headers=auth_headers)

    assert response.status_code == 200
    updated = mock_memory_store.update.call_args.args[0]
    assert updated.content == sample_memory_notes[0].content


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/api/memory"), ("put", f"/api/memory/{MEMORY_ID_1}")],
)
def test_authentication_precedes_body_validation_without_resolving_binding(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    resolver = Mock(side_effect=AssertionError("unauthenticated request must not bind"))
    set_trusted_memory_binding_resolver(resolver)

    response = client.request(
        method,
        path,
        content=b"",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    resolver.assert_not_called()


class TestMemoryListEndpoint:
    """Test cases for memory list endpoint."""

    def test_list_memories_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory listing."""
        # Setup mock response - memory API expects list_all to return a list-like object
        mock_memory_store.list_all.return_value = sample_memory_notes

        response = client.get("/api/memory/list", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total_count" in data
        assert "filters_used" in data
        assert data["total_count"] == 2
        assert len(data["memories"]) == 2

        # Verify memory structure
        memory = data["memories"][0]
        assert "id" in memory
        assert "content" in memory
        assert "keywords" in memory
        assert "tags" in memory
        assert "category" in memory
        assert "timestamp" in memory
        assert "metadata" in memory
        # Ensure context field is not present
        assert "context" not in memory

    def test_list_memories_with_category_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with category filter."""
        filtered_memories = [
            MemoryNote(
                id="filtered_1",
                content="Filtered memory",
                keywords=["test"],
                category="general",
                metadata={"source": "test"},
            )
        ]
        mock_memory_store.list_all.return_value = filtered_memories

        response = client.get("/api/memory/list?category=general", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 1
        assert data["memories"][0]["category"] == "general"

    def test_list_memories_with_tags_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with tags filter."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?tags=important,normal", headers=auth_headers
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "tags" in filters
        assert set(filters["tags"]) == {"important", "normal"}

    def test_list_memories_with_keywords_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with keywords filter."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?keywords=test,example", headers=auth_headers
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "keywords" in filters
        assert set(filters["keywords"]) == {"test", "example"}

    def test_list_memories_with_date_filters(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with date range filters."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?date_from=2024-01-01&date_to=2024-12-31",
            headers=auth_headers,
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "date_from" in filters
        assert "date_to" in filters

    def test_list_memories_with_limit_and_offset(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with pagination."""
        # Create more memories than limit
        many_memories = [
            MemoryNote(
                id=f"memory_{i}",
                content=f"Test memory {i}",
                keywords=["test"],
                category="general",
            )
            for i in range(10)
        ]
        mock_memory_store.list_all.return_value = many_memories

        response = client.get("/api/memory/list?limit=5&offset=3", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 5
        assert data["total_count"] == 10

    def test_list_memories_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory listing."""
        # Setup mock to raise exception
        mock_memory_store.list_all.side_effect = Exception("Database error")

        response = client.get("/api/memory/list", headers=auth_headers)

        assert response.status_code == 503
        assert response.json()["detail"] == "Memory storage is temporarily unavailable."
        assert "Database error" not in response.text

    def test_search_memories_returns_empty_success_for_no_match(
        self, client, mock_memory_store, auth_headers
    ):
        mock_memory_store.search.return_value = []

        response = client.get("/api/memory/list?search=missing", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["memories"] == []
        assert response.json()["total_count"] == 0

    def test_search_memories_sanitizes_backend_failure(
        self, client, mock_memory_store, auth_headers
    ):
        mock_memory_store.search.side_effect = RuntimeError(
            "lancedb://internal-host/private-table"
        )

        response = client.get(
            "/api/memory/list?search=preferences", headers=auth_headers
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Memory storage is temporarily unavailable."
        assert "internal-host" not in response.text


class TestMemoryGetEndpoint:
    """Test cases for memory get endpoint."""

    def test_get_memory_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory retrieval."""
        mock_memory_store.get.return_value = Mock(
            success=True, memory_id=MEMORY_ID_1, content=sample_memory_notes[0]
        )

        response = client.get(f"/api/memory/{MEMORY_ID_1}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == MEMORY_ID_1
        assert data["content"] == "Test memory 1"
        assert "context" not in data  # Ensure context field is not present

    def test_get_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory retrieval for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found", memory_id=MISSING_MEMORY_ID
        )

        response = client.get(f"/api/memory/{MISSING_MEMORY_ID}", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_memory_error_handling(self, client, mock_memory_store, auth_headers):
        """Test error handling in memory retrieval."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        response = client.get(f"/api/memory/{OTHER_MEMORY_ID}", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to get memory" in response.json()["detail"]


class TestMemoryCreateEndpoint:
    """Test cases for memory create endpoint."""

    def test_create_memory_success(self, client, mock_memory_store, auth_headers):
        """Test successful memory creation."""
        # Setup mock response
        mock_memory_store.add.return_value = Mock(
            success=True, memory_id="new_memory_123"
        )

        memory_data = {
            "content": "New test memory",
            "keywords": ["test", "new"],
            "tags": ["example"],
            "category": "general",
            "metadata": {"source": "test"},
        }

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["memory_id"] == "new_memory_123"
        assert "Memory created successfully" in data["message"]

        # Verify that add was called with MemoryNote
        mock_memory_store.add.assert_called_once()
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "New test memory"
        assert call_args.category == "general"

    def test_create_memory_with_defaults(self, client, mock_memory_store, auth_headers):
        """Test memory creation with default values."""
        mock_memory_store.add.return_value = Mock(
            success=True, memory_id="default_memory"
        )

        # Minimal memory data
        memory_data = {"content": "Minimal memory"}

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify default values were applied
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "Minimal memory"
        assert call_args.category == "general"  # Default category
        assert call_args.keywords == []  # Default empty list
        assert call_args.tags == []  # Default empty list
        assert call_args.metadata == {}  # Default empty dict

    @pytest.mark.parametrize("content", [None, "", " \t\n", 7])
    def test_create_rejects_non_string_or_blank_content_without_store_call(
        self, client, mock_memory_store, auth_headers, content
    ):
        response = client.post(
            "/api/memory/", json={"content": content}, headers=auth_headers
        )

        assert response.status_code == 422
        mock_memory_store.add.assert_not_called()

    def test_create_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory creation."""
        # Setup mock to return failure
        mock_memory_store.add.return_value = Mock(success=False, error="Storage failed")

        memory_data = {"content": "Test memory"}

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 500
        # The route raises ``response.error or "Failed to create memory"``, so
        # the store's own message is the intended detail. It used to be caught
        # by the route's own broad handler and re-wrapped, status code and all,
        # into "Failed to create memory: 500: Storage failed"; that wrapper is
        # gone because it also swallowed the lifecycle 503.
        assert response.json()["detail"] == "Storage failed"

    def test_create_memory_error_handling_without_a_store_message(
        self, client, mock_memory_store, auth_headers
    ):
        """A refusal that carries no message still answers the generic 500."""
        mock_memory_store.add.return_value = Mock(success=False, error=None)

        response = client.post(
            "/api/memory/", json={"content": "Test memory"}, headers=auth_headers
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to create memory"


class TestMemoryUpdateEndpoint:
    """Test cases for memory update endpoint."""

    def test_update_memory_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory update."""
        # Setup mock responses
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.return_value = Mock(success=True)

        update_data = {
            "content": "Updated content",
            "keywords": ["updated"],
            "tags": ["modified"],
            "category": "general",
            "metadata": {"source": "test", "updated": True},
        }

        response = client.put(
            f"/api/memory/{MEMORY_ID_1}", json=update_data, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Memory updated successfully" in data["message"]

        # Verify that update was called with updated MemoryNote
        mock_memory_store.update.assert_called_once()
        call_args = mock_memory_store.update.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "Updated content"
        assert call_args.keywords == ["updated"]
        assert call_args.id == MEMORY_ID_1

    def test_update_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory update for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found"
        )

        update_data = {"content": "Updated content"}

        response = client.put(
            f"/api/memory/{MISSING_MEMORY_ID}", json=update_data, headers=auth_headers
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_memory_partial_update(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test partial memory update."""
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.return_value = Mock(success=True)

        # Update only content
        update_data = {"content": "Only content updated"}

        response = client.put(
            f"/api/memory/{MEMORY_ID_1}", json=update_data, headers=auth_headers
        )

        assert response.status_code == 200

        # Verify that only content was updated, other fields preserved
        call_args = mock_memory_store.update.call_args[0][0]
        assert call_args.content == "Only content updated"
        assert call_args.keywords == ["test", "sample"]  # Original values preserved
        assert call_args.tags == ["important"]  # Original values preserved

    @pytest.mark.parametrize("content", ["", " \t\n"])
    def test_update_rejects_blank_content_without_store_call(
        self, client, mock_memory_store, auth_headers, content
    ):
        response = client.put(
            f"/api/memory/{MEMORY_ID_1}",
            json={"content": content},
            headers=auth_headers,
        )

        assert response.status_code == 422
        mock_memory_store.get.assert_not_called()
        mock_memory_store.update.assert_not_called()

    def test_update_preserves_nonblank_content_exactly(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            f"/api/memory/{MEMORY_ID_1}",
            json={"content": "  Mixed Case  "},
            headers=auth_headers,
        )

        assert response.status_code == 200
        updated = mock_memory_store.update.call_args.args[0]
        assert updated.content == "  Mixed Case  "

    def test_update_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory update."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        update_data = {"content": "Updated content"}

        response = client.put(
            f"/api/memory/{OTHER_MEMORY_ID}", json=update_data, headers=auth_headers
        )

        assert response.status_code == 500
        assert "Failed to update memory" in response.json()["detail"]


class TestMemoryDeleteEndpoint:
    """Test cases for memory delete endpoint."""

    def test_delete_memory_success(self, client, mock_memory_store, auth_headers):
        """Test successful memory deletion."""
        mock_memory_store.delete.return_value = Mock(success=True)

        response = client.delete(f"/api/memory/{MEMORY_ID_1}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Memory deleted successfully" in data["message"]

        # Verify delete was called with correct ID
        mock_memory_store.delete.assert_called_once_with(MEMORY_ID_1)

    def test_delete_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory deletion for non-existent ID."""
        mock_memory_store.delete.return_value = Mock(
            success=False, error="Memory not found"
        )

        response = client.delete(
            f"/api/memory/{MISSING_MEMORY_ID}", headers=auth_headers
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory deletion."""
        # Setup mock to raise exception
        mock_memory_store.delete.side_effect = Exception("Database error")

        response = client.delete(f"/api/memory/{OTHER_MEMORY_ID}", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to delete memory" in response.json()["detail"]


# Opaque IDs a store can hold: legacy admission keeps any unique non-empty
# string ID and ``LanceDBMemoryStore.add`` keeps any non-empty supplied ID.
OPAQUE_IDS = [
    "legacy-note-1",
    "memory_1",
    "a'b",
    "id with spaces",
    "50%_off",
    "back\\slash",
    "a/b",
    MEMORY_ID_1.upper(),
    # A prefix ID and the same ID with a trailing newline must stay distinct.
    "note",
    "note\n",
    "note\r\n",
]
# Values that used to rewrite the LanceDB predicate when interpolated raw.
HOSTILE_IDS = [
    "x' OR '1'='1",
    "') OR (1=1",
    "' OR true OR '",
    "'",
    "''",
    "\\",
    "\\' OR 1=1 --",
    "%",
    "_",
    MEMORY_ID_1 + "' OR '1'='1",
]


def _memory_path(memory_id: str) -> str:
    return "/api/memory/" + quote(memory_id, safe="")


class TestOpaqueMemoryIds:
    """By-ID routes pass the exact listed ID to the store, never a rewrite of it.

    Safety comes from the store quoting the ID as one SQL literal (see
    ``TestOpaqueMemoryIdsAgainstAdmittedLegacyStore`` and
    ``tests/core/memory/test_memory_id_filter_injection.py``), not from
    restricting the ID's shape at the API.
    """

    @pytest.mark.parametrize("memory_id", OPAQUE_IDS + HOSTILE_IDS)
    def test_get_passes_the_exact_id(
        self, client, mock_memory_store, auth_headers, memory_id
    ):
        mock_memory_store.get.return_value = Mock(
            success=True, content=MemoryNote(id=memory_id, content="x")
        )

        response = client.get(_memory_path(memory_id), headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["id"] == memory_id
        mock_memory_store.get.assert_called_once_with(memory_id)

    @pytest.mark.parametrize("memory_id", OPAQUE_IDS + HOSTILE_IDS)
    def test_update_passes_the_exact_id(
        self, client, mock_memory_store, auth_headers, memory_id
    ):
        mock_memory_store.get.return_value = Mock(
            success=True, content=MemoryNote(id=memory_id, content="x")
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            _memory_path(memory_id), json={"content": "y"}, headers=auth_headers
        )

        assert response.status_code == 200
        mock_memory_store.get.assert_called_once_with(memory_id)
        assert mock_memory_store.update.call_args[0][0].id == memory_id

    @pytest.mark.parametrize("memory_id", OPAQUE_IDS + HOSTILE_IDS)
    def test_delete_passes_the_exact_id(
        self, client, mock_memory_store, auth_headers, memory_id
    ):
        mock_memory_store.delete.return_value = Mock(success=True)

        response = client.delete(_memory_path(memory_id), headers=auth_headers)

        assert response.status_code == 200
        mock_memory_store.delete.assert_called_once_with(memory_id)

    @pytest.mark.parametrize("method", ["get", "put", "delete"])
    def test_empty_id_never_reaches_a_by_id_route(
        self, client, mock_memory_store, auth_headers, method
    ):
        # No store can hold an empty ID; ``/api/memory/`` stays the collection
        # path (POST-only), so it never reaches the store as an empty ID.
        kwargs = {"json": {"content": "y"}} if method == "put" else {}
        response = getattr(client, method)(
            "/api/memory/", headers=auth_headers, **kwargs
        )

        assert response.status_code == 405
        mock_memory_store.get.assert_not_called()
        mock_memory_store.update.assert_not_called()
        mock_memory_store.delete.assert_not_called()


class _FixedEmbedding(BaseEmbedding):
    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        if isinstance(text, str):
            return [0.1] * 4
        return [[0.1] * 4 for _ in text]

    def get_dimension(self) -> int:
        return 4

    @property
    def abilities(self) -> list[str]:
        return ["embed"]


class TestOpaqueMemoryIdsAgainstAdmittedLegacyStore:
    """List -> GET/PUT/DELETE round trip on a real, legacy-admitted LanceDB table.

    The table is seeded the way a pre-UUID deployment would have left it, with
    opaque and hostile IDs, admitted through production legacy admission and
    served through the production isolation wrapper. Every ID ``/list`` returns
    must work at all three by-ID routes for its owner, and only for its owner.
    """

    IDENTITY = EmbeddingIdentity("openai", "test-embedding", None, 4, None)
    VICTIM_ID = "victim-note"

    @pytest.fixture
    def store(self, tmp_path, test_db):
        owner = int(test_db.id)
        self.other_user = owner + 1000
        owned = OPAQUE_IDS + HOSTILE_IDS
        rows = [(memory_id, owner) for memory_id in owned]
        rows.append((self.VICTIM_ID, self.other_user))
        # The victim also owns notes whose IDs collide with hostile patterns'
        # intended matches, so any widened predicate would hit them.
        rows.append(("1", self.other_user))
        rows.append(("victim%", self.other_user))

        db_dir = str(tmp_path / "memory")
        connection = lancedb.connect(db_dir)
        table = connection.create_table(
            "memories",
            pa.table(
                {
                    "id": [memory_id for memory_id, _ in rows],
                    "text": [f"note {memory_id}" for memory_id, _ in rows],
                    "metadata": [
                        json.dumps({"content": f"note {memory_id}", "user_id": user_id})
                        for memory_id, user_id in rows
                    ],
                }
            ),
        )
        _safe_close_table(table)

        outcome = admit_lancedb_memory_storage(
            DormantLanceDBMemoryHandle(connection, "memories"),
            self.IDENTITY,
            writers_quiesced=True,
        )
        assert outcome.state is StorageAdmissionState.ADMITTED
        assert outcome.admitted is not None
        base = LanceDBMemoryStore.open_admitted(
            outcome.admitted,
            self.IDENTITY,
            db_dir=db_dir,
            embedding_model=_FixedEmbedding(),
        )
        self.base = base
        return UserIsolatedMemoryStore(base)

    @pytest.fixture
    def memory_router(self, store):
        return MemoryManagementRouter(lambda: store)

    def _other_users_notes(self) -> dict[str, str]:
        with UserContext(self.other_user):
            return {
                note.id: str(note.content)
                for note in self.base.list_all(
                    {"metadata": {"user_id": self.other_user}}
                )
            }

    def test_listed_ids_round_trip_through_get_put_delete(self, client, auth_headers):
        victims_before = self._other_users_notes()
        assert set(victims_before) == {self.VICTIM_ID, "1", "victim%"}

        listed = client.get("/api/memory/list", headers=auth_headers)
        assert listed.status_code == 200
        listed_ids = [memory["id"] for memory in listed.json()["memories"]]
        assert sorted(listed_ids) == sorted(OPAQUE_IDS + HOSTILE_IDS)

        for memory_id in listed_ids:
            path = _memory_path(memory_id)

            got = client.get(path, headers=auth_headers)
            assert got.status_code == 200, memory_id
            assert got.json()["id"] == memory_id
            assert got.json()["content"] == f"note {memory_id}"

            put = client.put(
                path, json={"content": f"edited {memory_id}"}, headers=auth_headers
            )
            assert put.status_code == 200, memory_id
            assert client.get(path, headers=auth_headers).json()["content"] == (
                f"edited {memory_id}"
            )

            deleted = client.delete(path, headers=auth_headers)
            assert deleted.status_code == 200, memory_id
            assert client.get(path, headers=auth_headers).status_code == 404

            # Each step touched exactly that one note: the other user's notes
            # are untouched after every hostile ID.
            assert self._other_users_notes() == victims_before, memory_id

        assert (
            client.get("/api/memory/list", headers=auth_headers).json()["memories"]
            == []
        )

    @pytest.mark.parametrize("victim_id", [VICTIM_ID, "1", "victim%"])
    def test_another_users_listed_id_is_not_reachable(
        self, client, auth_headers, victim_id
    ):
        victims_before = self._other_users_notes()
        path = _memory_path(victim_id)

        assert client.get(path, headers=auth_headers).status_code == 404
        assert (
            client.put(path, json={"content": "hijacked"}, headers=auth_headers)
        ).status_code == 404
        assert client.delete(path, headers=auth_headers).status_code == 404

        assert self._other_users_notes() == victims_before


class TestMemoryStatsEndpoint:
    """Test cases for memory stats endpoint."""

    def test_get_stats_success(self, client, mock_memory_store, auth_headers):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_count": 10,
            "category_counts": {"general": 5, "system": 3, "test": 2},
            "tag_counts": {"important": 3, "normal": 4, "test": 5},
            "memory_store_type": "in_memory",
        }
        mock_memory_store.get_stats.return_value = mock_stats

        response = client.get("/api/memory/stats", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 10
        assert data["category_counts"]["general"] == 5
        assert data["tag_counts"]["important"] == 3
        assert data["memory_store_type"] == "in_memory"

    def test_get_stats_with_error(self, client, mock_memory_store, auth_headers):
        """Test stats retrieval with error."""
        mock_stats = {
            "total_count": 0,
            "category_counts": {},
            "tag_counts": {},
            "memory_store_type": "in_memory",
            "error": "Database connection failed",
        }
        mock_memory_store.get_stats.return_value = mock_stats

        response = client.get("/api/memory/stats", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert "error" in data
        assert data["error"] == "Database connection failed"

    def test_get_stats_error_handling(self, client, mock_memory_store, auth_headers):
        """Test error handling in stats retrieval."""
        # Setup mock to raise exception
        mock_memory_store.get_stats.side_effect = Exception("Database error")

        response = client.get("/api/memory/stats", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to get memory stats" in response.json()["detail"]


class TestMemoryApiIntegration:
    """Integration tests for memory API endpoints."""

    def test_crd_cycle(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test Create-Read-Delete cycle."""

        # Setup mock responses
        def mock_add_side_effect(note):
            # Return the memory with the provided content
            return Mock(success=True, memory_id=CRD_MEMORY_ID)

        def mock_get_side_effect(memory_id):
            if memory_id == CRD_MEMORY_ID:
                # Create a memory note with the test data
                from xagent.core.memory.core import MemoryNote

                test_memory = MemoryNote(
                    id=CRD_MEMORY_ID,
                    content="CRD test memory",
                    keywords=["test", "crd"],
                    tags=[],
                    category="general",
                    metadata={},
                )
                return Mock(success=True, content=test_memory)
            return Mock(success=False, error="Memory not found")

        mock_memory_store.add.side_effect = mock_add_side_effect
        mock_memory_store.get.side_effect = mock_get_side_effect
        mock_memory_store.delete.return_value = Mock(success=True)

        # Create memory
        memory_data = {
            "content": "CRD test memory",
            "keywords": ["test", "crd"],
            "category": "general",
        }

        create_response = client.post(
            "/api/memory/", json=memory_data, headers=auth_headers
        )
        assert create_response.status_code == 200
        memory_id = create_response.json()["memory_id"]

        # Read memory
        get_response = client.get(f"/api/memory/{memory_id}", headers=auth_headers)
        assert get_response.status_code == 200
        assert get_response.json()["content"] == "CRD test memory"

        # Delete memory
        delete_response = client.delete(
            f"/api/memory/{memory_id}", headers=auth_headers
        )
        assert delete_response.status_code == 200

    def test_query_parameter_parsing(self, client, mock_memory_store, auth_headers):
        """Test that query parameters are parsed correctly."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?category=general&tags=tag1,tag2&keywords=kw1,kw2&limit=10&offset=5",
            headers=auth_headers,
        )

        assert response.status_code == 200

        # Verify that list_all was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}

        assert filters["category"] == "general"
        assert set(filters["tags"]) == {"tag1", "tag2"}
        assert set(filters["keywords"]) == {"kw1", "kw2"}

    def test_invalid_parameters(self, client, mock_memory_store, auth_headers):
        """Test handling of invalid parameters."""
        mock_memory_store.list_all.return_value = []

        # Test with invalid limit
        response = client.get("/api/memory/list?limit=invalid", headers=auth_headers)
        # Should still work due to FastAPI validation
        assert response.status_code == 422  # Validation error

    def test_content_type_validation(self, client, mock_memory_store, auth_headers):
        """Test content type validation for POST/PUT requests."""
        # Setup mock to return valid response for successful creation
        mock_memory_store.add.return_value = Mock(success=True, memory_id="test_123")

        # Test with invalid content type
        response = client.post(
            "/api/memory/", content="invalid content", headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

        # Test with valid JSON but invalid structure
        response = client.post(
            "/api/memory/", json={"invalid": "data"}, headers=auth_headers
        )
        # Should fail due to missing required content field
        assert response.status_code == 422


class TestMemoryApiUsage:
    """Memory API calls embed outside any agent run; the caller is billed."""

    @pytest.fixture
    def hook(self):
        from xagent.web.services import quota_hooks

        calls: list = []
        quota_hooks.set_standalone_usage_hook(
            lambda user_id, details, source: calls.append((user_id, details, source))
        )
        yield calls
        quota_hooks.set_standalone_usage_hook(None)

    @staticmethod
    def _embedding(result):
        from xagent.core.model.chat.token_context import (
            MediaCallType,
            add_media_usage,
        )

        def _call(*_args, **_kwargs):
            add_media_usage(
                call_type=MediaCallType.EMBEDDING, quantity=1, model_id="mem"
            )
            if isinstance(result, Exception):
                raise result
            return result

        return _call

    def _assert_billed(self, hook, test_db, rows=1):
        [(user_id, details, source)] = hook
        assert (user_id, source) == (test_db.id, "memory_api")
        assert [row["model_id"] for row in details] == ["mem"] * rows

    def test_search_bills_the_caller(
        self, client, mock_memory_store, auth_headers, hook, test_db
    ):
        mock_memory_store.search.side_effect = self._embedding([])
        response = client.get("/api/memory/list?search=q", headers=auth_headers)

        assert response.status_code == 200
        self._assert_billed(hook, test_db)

    def test_listing_without_search_bills_nothing(
        self, client, mock_memory_store, auth_headers, hook
    ):
        mock_memory_store.list_all.return_value = []
        client.get("/api/memory/list", headers=auth_headers)
        assert hook == []

    def test_create_bills_the_caller_even_when_the_store_fails(
        self, client, mock_memory_store, auth_headers, hook, test_db
    ):
        from xagent.core.memory.core import MemoryResponse

        mock_memory_store.add.side_effect = self._embedding(
            MemoryResponse(success=True, memory_id="m1")
        )
        assert (
            client.post(
                "/api/memory", json={"content": "x"}, headers=auth_headers
            ).status_code
            == 200
        )
        mock_memory_store.add.side_effect = self._embedding(RuntimeError("disk full"))
        assert (
            client.post(
                "/api/memory", json={"content": "x"}, headers=auth_headers
            ).status_code
            == 500
        )

        assert len(hook) == 2
        assert {call[0] for call in hook} == {test_db.id}

    def test_update_bills_the_caller(
        self,
        client,
        mock_memory_store,
        auth_headers,
        hook,
        test_db,
        sample_memory_notes,
    ):
        from xagent.core.memory.core import MemoryResponse

        mock_memory_store.get.return_value = MemoryResponse(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.side_effect = self._embedding(
            MemoryResponse(success=True)
        )
        response = client.put(
            f"/api/memory/{MEMORY_ID_1}", json={"content": "new"}, headers=auth_headers
        )

        assert response.status_code == 200
        self._assert_billed(hook, test_db)

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("get", "/api/memory/list?search=q", None),
            ("post", "/api/memory", {"content": "x"}),
            ("put", f"/api/memory/{MEMORY_ID_1}", {"content": "new"}),
        ],
        ids=["search", "create", "update"],
    )
    def test_a_gate_refusal_is_a_402_and_never_embeds(
        self,
        client,
        mock_memory_store,
        auth_headers,
        hook,
        test_db,
        sample_memory_notes,
        method,
        path,
        body,
    ):
        from xagent.core.memory.core import MemoryResponse
        from xagent.web.services import quota_hooks

        mock_memory_store.get.return_value = MemoryResponse(
            success=True, content=sample_memory_notes[0]
        )
        # The deployment's memory authority, under its billing id.
        mock_memory_store.embedding_usage_model_id = "platform/text-embedding-v4"
        gate_calls: list = []

        def gate(user_id, source, model_ids):
            gate_calls.append((user_id, source, model_ids))
            return "Out of AI credits"

        quota_hooks.set_standalone_gate_hook(gate)
        try:
            kwargs = {"headers": auth_headers}
            if body is not None:
                kwargs["json"] = body
            response = getattr(client, method)(path, **kwargs)
        finally:
            quota_hooks.set_standalone_gate_hook(None)

        assert response.status_code == 402
        assert response.json()["detail"]["code"] == "standalone_usage_refused"
        assert gate_calls == [
            (test_db.id, "memory_api", ("platform/text-embedding-v4",))
        ]
        mock_memory_store.search.assert_not_called()
        mock_memory_store.add.assert_not_called()
        mock_memory_store.update.assert_not_called()
        assert hook == []


@pytest.mark.parametrize(
    "billing_model_id,expected",
    [("platform/text-embedding-v4", "platform/text-embedding-v4"), ("", "")],
)
def test_a_lancedb_store_reports_the_id_its_embeddings_are_billed_under(
    billing_model_id, expected
):
    from xagent.core.memory.lancedb import LanceDBMemoryStore
    from xagent.core.model.embedding.adapter import create_embedding_adapter
    from xagent.core.model.model import EmbeddingModelConfig

    store = LanceDBMemoryStore.__new__(LanceDBMemoryStore)
    store._embedding_model = create_embedding_adapter(
        EmbeddingModelConfig(
            id="memory-authority",
            billing_model_id=billing_model_id,
            model_name="text-embedding-v4",
            model_provider="openai",
            api_key="test-key",
        )
    )
    assert store.embedding_usage_model_id == expected

    store._embedding_model = None
    assert store.embedding_usage_model_id == ""


class TestMemoryApiRunsStoreCallsOffTheEventLoop:
    """The store interface is synchronous; a blocking store must not stall the loop."""

    @pytest.fixture
    def blocking_app(self, mock_memory_store, test_db, auth_headers):
        import threading

        entered = threading.Event()
        release = threading.Event()
        seen: dict[str, Any] = {}

        def block(*_args, **_kwargs):
            from xagent.core.model.chat.token_context import token_context
            from xagent.core.user_context import current_user_id

            seen["user_id"] = current_user_id.get()
            seen["usage_bound"] = token_context.get(None) is not None
            entered.set()
            # Bounded so a regression fails the assertions instead of hanging.
            release.wait(timeout=10)
            return MemoryResponse(success=True, memory_id="m1")

        mock_memory_store.add.side_effect = block
        app = FastAPI()
        app.include_router(auth_router)
        app.include_router(
            MemoryManagementRouter(lambda: mock_memory_store).get_router()
        )

        @app.get("/api/unrelated/health")
        async def health() -> dict:
            return {"ok": True}

        app.dependency_overrides[get_db] = override_get_db
        return app, entered, release, seen, auth_headers

    async def _wait_entered(self, entered) -> None:
        import asyncio

        for _ in range(500):
            if entered.is_set():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the store call never started")

    async def test_unrelated_requests_progress_while_a_store_call_blocks(
        self, blocking_app, test_db
    ):
        import asyncio
        import threading

        import httpx

        app, entered, release, seen, headers = blocking_app
        # Frees the worker if the loop itself is wedged, so a regression fails
        # the assertions below rather than hanging the suite.
        watchdog = threading.Timer(3, release.set)
        watchdog.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                slow = asyncio.create_task(
                    client.post("/api/memory", json={"content": "x"}, headers=headers)
                )
                await self._wait_entered(entered)
                health = await asyncio.wait_for(
                    client.get("/api/unrelated/health"), timeout=1
                )
                still_blocked = not release.is_set()
                release.set()
                response = await slow
        finally:
            watchdog.cancel()
            release.set()

        assert health.status_code == 200
        assert still_blocked, "the event loop was stalled by the store call"
        assert response.status_code == 200
        # Context variables reach the worker thread.
        assert seen["user_id"] == test_db.id
        assert seen["usage_bound"] is True

    async def test_cancelled_request_waits_for_the_store_call_to_finish(
        self, blocking_app, mock_memory_store
    ):
        import asyncio
        import threading

        import httpx

        app, entered, release, _seen, headers = blocking_app
        finished = threading.Event()
        inner = mock_memory_store.add.side_effect

        def tracked(*args, **kwargs):
            try:
                return inner(*args, **kwargs)
            finally:
                finished.set()

        mock_memory_store.add.side_effect = tracked
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            slow = asyncio.create_task(
                client.post("/api/memory", json={"content": "x"}, headers=headers)
            )
            await self._wait_entered(entered)
            slow.cancel()
            await asyncio.sleep(0.1)
            # Cancellation is deferred, not abandoned: the write is still running
            # and the request is not yet reported as anything.
            assert not slow.done()
            assert not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await slow
        assert finished.is_set()

    def test_revocation_mid_operation_is_still_a_503_through_the_thread(
        self, client, mock_memory_store, auth_headers
    ):
        from xagent.web.memory_lifecycle import (
            MemoryLifecycleState,
            MemoryLifecycleStatus,
            MemoryUnavailableError,
        )

        status = MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)
        mock_memory_store.list_all.side_effect = MemoryUnavailableError(status)

        response = client.get("/api/memory/list", headers=auth_headers)

        assert response.status_code == 503
        assert response.json()["detail"] == status.detail
