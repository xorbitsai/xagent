"""Test model management API functionality"""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.model import model_router
from xagent.web.models.database import Base, get_db, get_engine

# Create temporary directory for database


def override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


# Create test app without startup events
test_app = FastAPI()
test_app.include_router(auth_router)
test_app.include_router(model_router)
test_app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(test_app)


def ensure_system_initialized() -> None:
    status_response = client.get("/api/auth/setup-status")
    assert status_response.status_code == 200
    status_data = status_response.json()

    if status_data.get("needs_setup", True):
        setup_response = client.post(
            "/api/auth/setup-admin", json={"username": "admin", "password": "admin123"}
        )
        assert setup_response.status_code == 200
        assert setup_response.json().get("success") is True


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    # Base.metadata.create_all(bind=engine)
    # Initialize database with default users
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"

    with patch("xagent.web.models.database.try_upgrade_db"):
        init_db(db_url=SQLALCHEMY_DATABASE_URL)

    engine = get_engine()

    yield

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="function")
def admin_user(test_db):
    """Create admin user for testing"""
    ensure_system_initialized()

    db = next(get_db())
    from xagent.web.models.user import User

    admin = db.query(User).filter(User.username == "admin").first()
    assert admin is not None
    user_info = {"id": admin.id, "username": admin.username}
    db.close()
    return user_info


@pytest.fixture(scope="function")
def regular_user(test_db):
    """Create regular user for testing"""
    ensure_system_initialized()

    user_data = {"username": "regularuser", "password": "password123"}
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 200
    assert response.json().get("success") is True
    return response.json()["user"]


@pytest.fixture(scope="function")
def admin_headers(admin_user):
    """Authentication headers for admin user"""
    response = client.post(
        "/api/auth/login",
        json={"username": admin_user["username"], "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="function")
def regular_headers(regular_user):
    """Authentication headers for regular user"""
    response = client.post(
        "/api/auth/login",
        json={"username": regular_user["username"], "password": "password123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="function")
def sample_model_data():
    """Sample model data for testing"""
    return {
        "model_id": "test-openai-model",
        "category": "llm",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "api_key": "test-api-key",
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.7,
        "abilities": ["chat", "tool_calling"],
        "description": "Test OpenAI model",
        "share_with_users": False,
    }


class TestModelAPI:
    """Test model management API endpoints"""

    def test_create_model_as_admin(
        self, test_db, admin_user, admin_headers, sample_model_data
    ):
        """Test model creation as admin user"""
        response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == sample_model_data["model_id"]
        assert data["category"] == sample_model_data["category"]
        assert data["model_provider"] == sample_model_data["model_provider"]
        assert data["is_owner"] is True
        assert data["can_edit"] is True
        assert data["can_delete"] is True
        assert data["is_shared"] is False

    def test_create_model_as_regular_user(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test model creation as regular user"""
        response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == sample_model_data["model_id"]
        assert data["is_owner"] is True
        assert data["can_edit"] is True
        assert data["can_delete"] is True
        assert data["is_shared"] is False

    def test_create_shared_model_as_admin(
        self, test_db, admin_user, admin_headers, sample_model_data
    ):
        """Test creating shared model as admin user"""
        sample_model_data["share_with_users"] = True
        response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_shared"] is True

    def test_create_shared_model_as_regular_user_fails(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test that regular user cannot create shared models"""
        sample_model_data["share_with_users"] = True
        response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response.status_code == 403
        data = response.json()
        assert "Only administrators can share models" in data["detail"]

    def test_get_user_models(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test getting user's models"""
        # Create a model first
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200

        # Get user models
        response = client.get("/api/models/", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_id"] == sample_model_data["model_id"]

    def test_get_shared_models(
        self,
        test_db,
        admin_user,
        admin_headers,
        regular_user,
        regular_headers,
        sample_model_data,
    ):
        """Test getting shared models"""
        # Admin creates a shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # Regular user should see the shared model
        response = client.get("/api/models/", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model_id"] == sample_model_data["model_id"]
        assert data[0]["is_shared"] is True
        assert data[0]["is_owner"] is False  # Regular user is not owner
        assert data[0]["can_edit"] is False  # Regular user cannot edit
        assert data[0]["can_delete"] is False  # Regular user cannot delete

    def test_get_model_by_id(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test getting specific model by ID"""
        # Create a model first
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_id_int = create_response.json()["id"]

        # Get model by string model_id (as expected by API)
        response = client.get(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model_id_int
        assert data["model_id"] == sample_model_data["model_id"]

    def test_update_model_as_owner(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test updating model as owner"""
        # Create a model first
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Update model
        update_data = {"temperature": 0.8, "description": "Updated description"}
        response = client.put(
            f"/api/models/{model_id_str}", json=update_data, headers=regular_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.8
        assert data["description"] == "Updated description"

    def test_update_shared_model_as_non_owner_fails(
        self,
        test_db,
        admin_user,
        admin_headers,
        regular_user,
        regular_headers,
        sample_model_data,
    ):
        """Test that non-owner cannot update shared model"""
        # Admin creates a shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Regular user tries to update
        update_data = {"temperature": 0.8}
        response = client.put(
            f"/api/models/{model_id_str}", json=update_data, headers=regular_headers
        )
        assert response.status_code == 403
        data = response.json()
        assert "No permission to edit this model" in data["detail"]

    def test_delete_model_as_owner(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test deleting model as owner"""
        # Create a model first
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Delete model
        response = client.delete(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Model deleted successfully"

        # Verify model is deleted
        get_response = client.get(
            f"/api/models/{model_id_str}", headers=regular_headers
        )
        assert get_response.status_code == 404

    def test_delete_shared_model_as_non_owner_fails(
        self,
        test_db,
        admin_user,
        admin_headers,
        regular_user,
        regular_headers,
        sample_model_data,
    ):
        """Test that non-owner cannot delete shared model"""
        # Admin creates a shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Regular user tries to delete
        response = client.delete(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 403
        data = response.json()
        assert "No permission to delete this model" in data["detail"]

    def test_get_nonexistent_model(self, test_db, regular_user, regular_headers):
        """Test getting non-existent model"""
        response = client.get("/api/models/99999", headers=regular_headers)
        assert response.status_code == 404

    def test_update_nonexistent_model(self, test_db, regular_user, regular_headers):
        """Test updating non-existent model"""
        update_data = {"temperature": 0.8}
        response = client.put(
            "/api/models/99999", json=update_data, headers=regular_headers
        )
        assert response.status_code == 404

    def test_delete_nonexistent_model(self, test_db, regular_user, regular_headers):
        """Test deleting non-existent model"""
        response = client.delete("/api/models/99999", headers=regular_headers)
        assert response.status_code == 404

    def test_create_model_with_missing_fields(
        self, test_db, regular_user, regular_headers
    ):
        """Test creating model with missing required fields"""
        incomplete_data = {
            "model_id": "test-model"
            # Missing required fields
        }
        response = client.post(
            "/api/models/", json=incomplete_data, headers=regular_headers
        )
        assert response.status_code == 422  # Validation error

    def test_create_duplicate_model_id(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test creating model with duplicate model_id"""
        # Create first model
        response1 = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response1.status_code == 200

        # Try to create model with same model_id
        response2 = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response2.status_code == 400
        data = response2.json()
        assert "Model ID already exists" in data["detail"]

    def test_user_model_isolation(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test that users only see their own models and shared models"""
        # Create a model as regular user
        response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response.status_code == 200

        # Create another user and verify they don't see the first user's private model
        user2_data = {"username": "user2", "password": "password2"}
        user2_response = client.post("/api/auth/register", json=user2_data)
        assert user2_response.status_code == 200
        assert user2_response.json().get("success") is True

        # User2 should not see user1's private model
        models_response = client.get("/api/models/", headers=regular_headers)
        assert models_response.status_code == 200
        data = models_response.json()
        assert len(data) == 1  # User should still see their own model

    def test_model_with_abilities(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test creating model with abilities"""
        sample_model_data["abilities"] = ["chat", "tool_calling", "vision"]
        response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["abilities"] == ["chat", "tool_calling", "vision"]
