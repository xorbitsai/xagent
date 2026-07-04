"""Test model management API functionality"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import quote

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
            "/api/auth/setup-admin",
            json={
                "username": "admin",
                "email": "admin@example.com",
                "password": "admin123",
            },
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

    # Note: Previously mocked try_upgrade_db to skip db migrations.
    # Now removed the mock to test the complete init_db() flow.
    # For new databases (like this temp one), try_upgrade_db only stamps
    # the latest revision without running migrations, which is safe and correct.
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

    user_data = {
        "username": "regularuser",
        "email": "regularuser@example.com",
        "password": "password123",
    }
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


@pytest.fixture(scope="function")
def sample_embedding_model_data():
    return {
        "model_id": "test-embedding-model",
        "category": "embedding",
        "model_provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_key": "test-api-key",
        "base_url": "https://api.openai.com/v1",
        "dimension": 1536,
        "abilities": ["embedding"],
        "description": "Test embedding model",
        "share_with_users": False,
    }


@pytest.fixture(scope="function")
def sample_speech_model_data():
    return {
        "model_id": "test-speech-model",
        "category": "speech",
        "model_provider": "xinference",
        "model_name": "speech-dual-model",
        "api_key": "test-api-key",
        "base_url": "http://localhost:9997",
        "abilities": ["asr", "tts"],
        "description": "Test speech model",
        "share_with_users": False,
    }


@pytest.fixture(scope="function")
def sample_image_model_data():
    return {
        "model_id": "test-image-model",
        "category": "image",
        "model_provider": "dashscope",
        "model_name": "qwen-image",
        "api_key": "test-api-key",
        "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        "abilities": ["generate"],
        "description": "Test image model",
        "share_with_users": False,
    }


@pytest.fixture(scope="function")
def sample_video_model_data():
    return {
        "model_id": "test-video-model",
        "category": "video",
        "model_provider": "volcengine-ark",
        "model_name": "doubao-seedance-2-0-fast-260128",
        "api_key": "test-api-key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "abilities": ["generate"],
        "description": "Test Seedance video model",
        "share_with_users": False,
    }


class TestModelAPI:
    """Test model management API endpoints"""

    def test_test_connection_embedding_uses_embedding_adapter(
        self, test_db, regular_user, regular_headers
    ):
        """Embedding connection tests should use the embedding adapter instead of chat-only adapter."""
        embedding_model = Mock()
        embedding_model.encode = Mock(return_value=[0.1, 0.2, 0.3])

        with patch(
            "xagent.core.model.embedding.adapter.create_embedding_adapter",
            return_value=embedding_model,
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "openai",
                    "model_name": "text-embedding-3-small",
                    "api_key": "test-api-key",
                    "base_url": "https://api.openai.com/v1",
                    "category": "embedding",
                    "dimension": 1536,
                    "abilities": ["embedding"],
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        embedding_model.encode.assert_called_once_with("hello")

    def test_test_connection_image_fails_when_requested_ability_is_unsupported(
        self, test_db, regular_user, regular_headers
    ):
        """Image connection tests should fail instead of returning a false positive."""
        with (
            patch(
                "xagent.core.model.image.adapter.create_image_model",
                return_value=Mock(),
            ),
            patch(
                "xagent.web.services.model_list_service.fetch_models_from_provider",
                new=AsyncMock(
                    return_value=[
                        {
                            "id": "qwen-image",
                            "abilities": ["generate"],
                        }
                    ]
                ),
            ),
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "dashscope",
                    "model_name": "qwen-image",
                    "api_key": "test-api-key",
                    "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                    "category": "image",
                    "abilities": ["edit"],
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "does not support abilities: edit" in data["error"]

    def test_test_connection_speech_xinference_supports_empty_api_key(
        self, test_db, regular_user, regular_headers
    ):
        """Speech connection tests should work for Xinference without requiring an API key."""
        with (
            patch(
                "xagent.web.services.model_list_service.fetch_models_from_provider",
                new=AsyncMock(
                    return_value=[
                        {
                            "id": "whisper-base",
                            "abilities": ["asr"],
                        }
                    ]
                ),
            ),
            patch(
                "xagent.core.model.xinference_base.BaseXinferenceModel._ensure_model_handle",
                new=AsyncMock(return_value=object()),
            ),
            patch(
                "xagent.core.model.xinference_base.BaseXinferenceModel.aclose",
                new=AsyncMock(return_value=None),
            ),
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "xinference",
                    "model_name": "whisper-base",
                    "api_key": "",
                    "base_url": "http://localhost:9997",
                    "category": "speech",
                    "abilities": ["asr"],
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"

    def test_test_connection_speech_elevenlabs_uses_model_listing(
        self, test_db, regular_user, regular_headers
    ):
        """ElevenLabs TTS connection checks should not synthesize paid audio."""
        with patch(
            "xagent.web.services.model_list_service.fetch_models_from_provider",
            new=AsyncMock(
                return_value=[
                    {
                        "id": "eleven_v3",
                        "abilities": ["tts"],
                    }
                ]
            ),
        ) as mock_fetch:
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "elevenlabs",
                    "model_name": "eleven_v3",
                    "api_key": "test-api-key",
                    "category": "speech",
                    "abilities": ["tts"],
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        mock_fetch.assert_awaited_once()

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

    def test_get_model_by_path_with_slash_id(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test getting a model whose model_id contains a slash."""
        sample_model_data["model_id"] = "google/gemini-2.5-flash"

        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200

        response = client.get(
            f"/api/models/by-id/{quote(sample_model_data['model_id'], safe='')}",
            headers=regular_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == sample_model_data["model_id"]

    def test_update_model_by_path_with_slash_id(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test updating a model whose model_id contains a slash."""
        sample_model_data["model_id"] = "google/gemini-2.5-flash"

        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200

        response = client.put(
            f"/api/models/by-id/{quote(sample_model_data['model_id'], safe='')}",
            json={"temperature": 0.5, "description": "Slash-safe update"},
            headers=regular_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.5
        assert data["description"] == "Slash-safe update"

    def test_delete_model_by_path_with_slash_id(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        """Test deleting a model whose model_id contains a slash."""
        sample_model_data["model_id"] = "google/gemini-2.5-flash"

        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        assert create_response.status_code == 200

        delete_response = client.delete(
            f"/api/models/by-id/{quote(sample_model_data['model_id'], safe='')}",
            headers=regular_headers,
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["message"] == "Model deleted successfully"

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

        # Create another user
        user2_data = {
            "username": "user2",
            "email": "user2@example.com",
            "password": "password2",
        }
        user2_response = client.post("/api/auth/register", json=user2_data)
        assert user2_response.status_code == 200

        # Login as user2
        login2 = client.post(
            "/api/auth/login", json={"username": "user2", "password": "password2"}
        )
        user2_headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}

        # User2 should not see user1's private model
        models_response = client.get("/api/models/", headers=user2_headers)
        assert models_response.status_code == 200
        data = models_response.json()
        assert len(data) == 0  # User2 shouldn't see user1's private model

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

    def test_reject_embedding_model_as_general_default(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_embedding_model_data,
    ):
        create_response = client.post(
            "/api/models/",
            json=sample_embedding_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        embedding_model_id = create_response.json()["id"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": embedding_model_id, "config_type": "general"},
            headers=regular_headers,
        )

        assert default_response.status_code == 400
        assert "incompatible" in default_response.json()["detail"]

    def test_allow_embedding_model_as_embedding_default(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_embedding_model_data,
    ):
        create_response = client.post(
            "/api/models/",
            json=sample_embedding_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        embedding_model_id = create_response.json()["id"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": embedding_model_id, "config_type": "embedding"},
            headers=regular_headers,
        )

        assert default_response.status_code == 200
        assert default_response.json()["config_type"] == "embedding"

    def test_allow_dual_speech_model_for_individual_asr_and_tts_defaults(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_speech_model_data,
    ):
        create_response = client.post(
            "/api/models/",
            json=sample_speech_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        speech_model_id = create_response.json()["id"]

        asr_response = client.post(
            "/api/models/user-default",
            json={"model_id": speech_model_id, "config_type": "asr"},
            headers=regular_headers,
        )
        assert asr_response.status_code == 200
        assert asr_response.json()["config_type"] == "asr"

        tts_response = client.post(
            "/api/models/user-default",
            json={"model_id": speech_model_id, "config_type": "tts"},
            headers=regular_headers,
        )
        assert tts_response.status_code == 200
        assert tts_response.json()["config_type"] == "tts"

        defaults_response = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_response.status_code == 200
        defaults = {
            item["config_type"]: item["model"]["model_id"]
            for item in defaults_response.json()
        }
        assert defaults["asr"] == sample_speech_model_data["model_id"]
        assert defaults["tts"] == sample_speech_model_data["model_id"]

    def test_reject_asr_only_speech_model_as_tts_default(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_speech_model_data,
    ):
        sample_speech_model_data["abilities"] = ["asr"]
        create_response = client.post(
            "/api/models/",
            json=sample_speech_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        speech_model_id = create_response.json()["id"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": speech_model_id, "config_type": "tts"},
            headers=regular_headers,
        )

        assert default_response.status_code == 400
        assert "incompatible" in default_response.json()["detail"]

    def test_reject_generate_only_image_model_as_image_edit_default(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_image_model_data,
    ):
        create_response = client.post(
            "/api/models/",
            json=sample_image_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        image_model_id = create_response.json()["id"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": image_model_id, "config_type": "image_edit"},
            headers=regular_headers,
        )

        assert default_response.status_code == 400
        assert "incompatible" in default_response.json()["detail"]

    def test_create_video_model_and_set_default(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_video_model_data,
    ):
        create_response = client.post(
            "/api/models/",
            json=sample_video_model_data,
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        video_model = create_response.json()
        assert video_model["category"] == "video"
        assert video_model["model_provider"] == "volcengine-ark"
        assert video_model["abilities"] == ["generate"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": video_model["id"], "config_type": "video"},
            headers=regular_headers,
        )

        assert default_response.status_code == 200
        assert default_response.json()["config_type"] == "video"

    def test_dreamina_video_model_defaults_to_byteplus_base_url(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_video_model_data,
    ):
        sample_video_model_data["model_id"] = "test-dreamina-video-model"
        sample_video_model_data["model_provider"] = "byteplus-ark"
        sample_video_model_data["model_name"] = "dreamina-seedance-2-0-fast-260128"
        sample_video_model_data.pop("base_url")

        create_response = client.post(
            "/api/models/",
            json=sample_video_model_data,
            headers=regular_headers,
        )

        assert create_response.status_code == 200
        assert (
            create_response.json()["base_url"]
            == "https://ark.ap-southeast.bytepluses.com/api/v3"
        )

    def test_list_supported_providers_includes_deepseek(
        self, test_db, regular_user, regular_headers
    ):
        response = client.get(
            "/api/models/providers/supported",
            headers=regular_headers,
        )

        assert response.status_code == 200
        providers = response.json()["providers"]
        deepseek = next(
            (provider for provider in providers if provider["id"] == "deepseek"),
            None,
        )
        assert deepseek is not None
        assert deepseek["name"] == "DeepSeek"
        assert deepseek["default_base_url"] == "https://api.deepseek.com"

    def test_list_supported_providers_includes_ark_platforms(
        self, test_db, regular_user, regular_headers
    ):
        response = client.get(
            "/api/models/providers/supported",
            headers=regular_headers,
        )

        assert response.status_code == 200
        providers = response.json()["providers"]
        volcengine = next(
            (provider for provider in providers if provider["id"] == "volcengine-ark"),
            None,
        )
        byteplus = next(
            (provider for provider in providers if provider["id"] == "byteplus-ark"),
            None,
        )
        assert volcengine is not None
        assert volcengine["name"] == "Volcengine Ark"
        assert (
            volcengine["default_base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
        )
        assert byteplus is not None
        assert byteplus["name"] == "BytePlus Ark"
        assert (
            byteplus["default_base_url"]
            == "https://ark.ap-southeast.bytepluses.com/api/v3"
        )

    def test_fetch_deepseek_provider_models_returns_curated_v4_models(
        self, test_db, regular_user, regular_headers
    ):
        response = client.post(
            "/api/models/providers/deepseek/models",
            json={"api_key": "test-api-key"},
            headers=regular_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert [model["id"] for model in data["models"]] == [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ]
        assert data["models"][0]["abilities"] == [
            "chat",
            "tool_calling",
            "thinking_mode",
        ]

    def test_fetch_ark_provider_models_returns_platform_scoped_seedance(
        self, test_db, regular_user, regular_headers, monkeypatch
    ):
        async def fake_fetch_openai_models(api_key, base_url):
            if "bytepluses.com" in base_url:
                return [
                    {"id": "dreamina-seedance-2-0-fast-260128"},
                    {"id": "dreamina-seedream-4-0"},
                ]
            return [
                {"id": "doubao-seedance-1-5-pro-251215"},
                {"id": "doubao-seedance-2-0-fast-260128"},
                {"id": "doubao-1-5-pro-32k-250115"},
            ]

        monkeypatch.setattr(
            "xagent.web.services.model_list_service.fetch_openai_models",
            fake_fetch_openai_models,
        )

        domestic_response = client.post(
            "/api/models/providers/volcengine-ark/models",
            json={"api_key": "test-api-key", "category": "video"},
            headers=regular_headers,
        )

        assert domestic_response.status_code == 200
        domestic_data = domestic_response.json()
        domestic_models = {model["id"]: model for model in domestic_data["models"]}
        assert "doubao-seedance-1-5-pro-251215" in domestic_models
        assert "doubao-seedance-2-0-fast-260128" in domestic_models
        assert "doubao-1-5-pro-32k-250115" not in domestic_models
        assert "dreamina-seedance-2-0-fast-260128" not in domestic_models
        assert (
            domestic_models["doubao-seedance-2-0-fast-260128"]["default_base_url"]
            == "https://ark.cn-beijing.volces.com/api/v3"
        )

        byteplus_response = client.post(
            "/api/models/providers/byteplus-ark/models",
            json={"api_key": "test-api-key", "category": "video"},
            headers=regular_headers,
        )

        assert byteplus_response.status_code == 200
        byteplus_data = byteplus_response.json()
        byteplus_models = {model["id"]: model for model in byteplus_data["models"]}
        assert "dreamina-seedance-2-0-fast-260128" in byteplus_models
        assert "dreamina-seedream-4-0" not in byteplus_models
        assert "doubao-seedance-2-0-fast-260128" not in byteplus_models
        assert (
            byteplus_models["dreamina-seedance-2-0-fast-260128"]["default_base_url"]
            == "https://ark.ap-southeast.bytepluses.com/api/v3"
        )
        assert (
            byteplus_models["dreamina-seedance-2-0-fast-260128"]["category"] == "video"
        )
        assert byteplus_models["dreamina-seedance-2-0-fast-260128"]["abilities"] == [
            "generate"
        ]

    def test_fetch_xinference_video_provider_models(
        self, test_db, regular_user, regular_headers, monkeypatch
    ):
        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def json(self):
                return {
                    "data": [
                        {
                            "id": "wan-video",
                            "model_name": "Wan2.1-1.3B",
                            "model_type": "video",
                            "model_ability": ["text_to_video"],
                        },
                        {
                            "id": "chat-model",
                            "model_name": "qwen",
                            "model_type": "LLM",
                            "model_ability": ["chat"],
                        },
                    ]
                }

        class FakeClientSession:
            created_timeouts = []

            def __init__(self, *args, **kwargs):
                _ = args
                self.created_timeouts.append(kwargs.get("timeout"))

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def get(self, url, headers=None):
                assert url == "http://localhost:9997/v1/models"
                assert headers == {}
                return FakeResponse()

        monkeypatch.setattr(
            "xagent.web.services.model_list_service.aiohttp.ClientSession",
            FakeClientSession,
        )

        response = client.post(
            "/api/models/providers/xinference/models",
            json={
                "api_key": "",
                "base_url": "http://localhost:9997",
                "category": "video",
            },
            headers=regular_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["models"] == [
            {
                "id": "Wan2.1-1.3B",
                "model_uid": "wan-video",
                "model_type": "video",
                "category": "video",
                "model_ability": ["generate"],
                "abilities": ["generate"],
                "description": "",
            }
        ]
        assert FakeClientSession.created_timeouts[0].total == 30.0

    def test_fetch_xinference_video_provider_models_handles_unexpected_response(
        self, test_db, regular_user, regular_headers, monkeypatch
    ):
        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def json(self):
                return []

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                _ = args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def get(self, url, headers=None):
                assert url == "http://localhost:9997/v1/models"
                assert headers == {}
                return FakeResponse()

        monkeypatch.setattr(
            "xagent.web.services.model_list_service.aiohttp.ClientSession",
            FakeClientSession,
        )

        response = client.post(
            "/api/models/providers/xinference/models",
            json={
                "api_key": "",
                "base_url": "http://localhost:9997",
                "category": "video",
            },
            headers=regular_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["models"] == []

    def test_create_deepseek_rejects_legacy_alias(
        self, test_db, regular_user, regular_headers
    ):
        response = client.post(
            "/api/models/",
            json={
                "model_id": "legacy-deepseek-chat",
                "category": "llm",
                "model_provider": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "test-api-key",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
            headers=regular_headers,
        )

        assert response.status_code == 400
        assert "Unsupported DeepSeek model" in response.json()["detail"]

    def test_update_deepseek_rejects_legacy_alias(
        self, test_db, regular_user, regular_headers
    ):
        create_response = client.post(
            "/api/models/",
            json={
                "model_id": "valid-deepseek",
                "category": "llm",
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "api_key": "test-api-key",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
            headers=regular_headers,
        )
        assert create_response.status_code == 200

        response = client.put(
            "/api/models/valid-deepseek",
            json={"model_name": "deepseek-reasoner"},
            headers=regular_headers,
        )

        assert response.status_code == 400
        assert "Unsupported DeepSeek model" in response.json()["detail"]

    def test_test_connection_deepseek_disables_thinking(
        self, test_db, regular_user, regular_headers
    ):
        mock_llm = Mock()
        mock_llm.chat = AsyncMock(
            return_value={"type": "text", "content": "ok", "raw": {}}
        )

        with patch(
            "xagent.core.model.chat.basic.adapter.create_base_llm",
            return_value=mock_llm,
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "deepseek",
                    "model_name": "deepseek-v4-flash",
                    "api_key": "test-api-key",
                    "category": "llm",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        assert response.json()["status"] == "passed"
        mock_llm.chat.assert_awaited_once_with(
            [{"role": "user", "content": "Hello"}],
            max_tokens=16,
            thinking={"type": "disabled"},
        )
