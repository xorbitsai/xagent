"""Test cases for memory API endpoints."""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote
from xagent.web.api.auth import auth_router, hash_password
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User

# Create temporary directory for database
temp_dir = tempfile.mkdtemp()
temp_db_path = os.path.join(temp_dir, "test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": "admin",
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
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
            id="memory_1",
            content="Test memory 1",
            keywords=["test", "sample"],
            tags=["important"],
            category="general",
            metadata={"source": "test"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
        MemoryNote(
            id="memory_2",
            content="Test memory 2",
            keywords=["test", "example"],
            tags=["normal"],
            category="system",
            metadata={"source": "system"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
    ]


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

        assert response.status_code == 500
        assert "Failed to list memories" in response.json()["detail"]


class TestMemoryGetEndpoint:
    """Test cases for memory get endpoint."""

    def test_get_memory_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory retrieval."""
        mock_memory_store.get.return_value = Mock(
            success=True, memory_id="memory_1", content=sample_memory_notes[0]
        )

        response = client.get("/api/memory/memory_1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "memory_1"
        assert data["content"] == "Test memory 1"
        assert "context" not in data  # Ensure context field is not present

    def test_get_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory retrieval for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found", memory_id="non_existent"
        )

        response = client.get("/api/memory/non_existent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_memory_error_handling(self, client, mock_memory_store, auth_headers):
        """Test error handling in memory retrieval."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        response = client.get("/api/memory/test_id", headers=auth_headers)

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

    def test_create_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory creation."""
        # Setup mock to return failure
        mock_memory_store.add.return_value = Mock(success=False, error="Storage failed")

        memory_data = {"content": "Test memory"}

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to create memory" in response.json()["detail"]


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
            "/api/memory/memory_1", json=update_data, headers=auth_headers
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
        assert call_args.id == "memory_1"

    def test_update_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory update for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found"
        )

        update_data = {"content": "Updated content"}

        response = client.put(
            "/api/memory/non_existent", json=update_data, headers=auth_headers
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
            "/api/memory/memory_1", json=update_data, headers=auth_headers
        )

        assert response.status_code == 200

        # Verify that only content was updated, other fields preserved
        call_args = mock_memory_store.update.call_args[0][0]
        assert call_args.content == "Only content updated"
        assert call_args.keywords == ["test", "sample"]  # Original values preserved
        assert call_args.tags == ["important"]  # Original values preserved

    def test_update_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory update."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        update_data = {"content": "Updated content"}

        response = client.put(
            "/api/memory/test_id", json=update_data, headers=auth_headers
        )

        assert response.status_code == 500
        assert "Failed to update memory" in response.json()["detail"]


class TestMemoryDeleteEndpoint:
    """Test cases for memory delete endpoint."""

    def test_delete_memory_success(self, client, mock_memory_store, auth_headers):
        """Test successful memory deletion."""
        mock_memory_store.delete.return_value = Mock(success=True)

        response = client.delete("/api/memory/memory_1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Memory deleted successfully" in data["message"]

        # Verify delete was called with correct ID
        mock_memory_store.delete.assert_called_once_with("memory_1")

    def test_delete_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory deletion for non-existent ID."""
        mock_memory_store.delete.return_value = Mock(
            success=False, error="Memory not found"
        )

        response = client.delete("/api/memory/non_existent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory deletion."""
        # Setup mock to raise exception
        mock_memory_store.delete.side_effect = Exception("Database error")

        response = client.delete("/api/memory/test_id", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to delete memory" in response.json()["detail"]


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
            return Mock(success=True, memory_id="test_memory_123")

        def mock_get_side_effect(memory_id):
            if memory_id == "test_memory_123":
                # Create a memory note with the test data
                from xagent.core.memory.core import MemoryNote

                test_memory = MemoryNote(
                    id="test_memory_123",
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
