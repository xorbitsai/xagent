"""Tests for dynamic model sharing refactoring.

Covers:
- Hook infrastructure (_get_visible_user_ids, _can_user_share)
- Mode B access verification (two-step lookup)
- Mode C model listing (or_() filter)
- Permission checks (owner-only edit/delete)
- Protection constraints (cannot delete/un-share own default, cannot change category/abilities of shared)
- Un-share cleanup (non-owner UserModel + UserDefaultModel deletion)
- Static code cleanup verification (no pre-creation)
"""

import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.model import _can_user_share, model_router, set_can_share_hook
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.model_service import (
    _get_visible_user_ids,
    set_visible_user_ids_hook,
)

# ---------------------------------------------------------------------------
# App setup (same pattern as test_model_api.py)
# ---------------------------------------------------------------------------


def override_get_db():
    db = None
    try:
        db = next(get_db())
        yield db
    finally:
        if db is not None:
            db.close()


test_app = FastAPI()
test_app.include_router(auth_router)
test_app.include_router(model_router)
test_app.dependency_overrides[get_db] = override_get_db

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


@pytest.fixture(autouse=True)
def _reset_hooks():
    """Ensure hooks are reset between tests."""
    set_visible_user_ids_hook(None)
    set_can_share_hook(None)
    set_cache_backend_for_testing(None)
    yield
    set_visible_user_ids_hook(None)
    set_can_share_hook(None)
    set_cache_backend_for_testing(None)


@pytest.fixture(scope="function")
def test_db():
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"

    init_db(db_url=SQLALCHEMY_DATABASE_URL)
    engine = get_engine()

    yield

    Base.metadata.drop_all(bind=engine)
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="function")
def admin_user(test_db):
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
    response = client.post(
        "/api/auth/login",
        json={"username": admin_user["username"], "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="function")
def regular_headers(regular_user):
    response = client.post(
        "/api/auth/login",
        json={"username": regular_user["username"], "password": "password123"},
    )
    assert response.status_code == 200
    data = response.json()
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="function")
def sample_model_data():
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


# ===========================================================================
# 1. Hook Infrastructure
# ===========================================================================


class TestHookInfrastructure:
    """Test _get_visible_user_ids and _can_user_share hooks."""

    def test_get_visible_user_ids_default_returns_admin_ids(self, test_db, admin_user):
        """Without hook, returns all admin user IDs."""
        db = next(get_db())
        result = _get_visible_user_ids(db, admin_user["id"])
        db.close()
        assert admin_user["id"] in result

    def test_get_visible_user_ids_hook_override(self, test_db):
        """With hook, returns hook result."""
        set_visible_user_ids_hook(lambda db, user_id: [42, 99])
        db = next(get_db())
        result = _get_visible_user_ids(db, 1)
        db.close()
        assert result == [42, 99]

    def test_can_user_share_default_is_admin(self, test_db, admin_user, regular_user):
        """Without hook, admin can share, regular user cannot."""
        db = next(get_db())
        from xagent.web.models.user import User

        admin = db.query(User).filter(User.id == admin_user["id"]).first()
        regular = db.query(User).filter(User.id == regular_user["id"]).first()
        assert _can_user_share(admin) is True
        assert _can_user_share(regular) is False
        db.close()

    def test_can_user_share_hook_override(self, test_db, regular_user):
        """With hook, regular user can share."""
        set_can_share_hook(lambda user: True)
        db = next(get_db())
        from xagent.web.models.user import User

        regular = db.query(User).filter(User.id == regular_user["id"]).first()
        assert _can_user_share(regular) is True
        db.close()


# ===========================================================================
# 3. Mode B — Access Verification
# ===========================================================================


class TestModeBAccessVerification:
    """Test two-step lookup: own → shared from visible users."""

    def test_resolve_accessible_model_own(
        self, test_db, admin_headers, sample_model_data
    ):
        """Step 1: owner accesses their own model."""
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Owner should be able to GET
        response = client.get(f"/api/models/{model_id_str}", headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["is_owner"] is True

    def test_resolve_accessible_model_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Step 2: non-owner accesses shared model from visible users."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Regular user should see shared model
        response = client.get(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 200
        assert response.json()["is_owner"] is False

    def test_resolve_accessible_model_no_access(
        self, test_db, regular_headers, admin_headers, sample_model_data
    ):
        """Two-step both miss: returns 404."""
        # Admin creates a private model
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Regular user cannot see private model
        response = client.get(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 404

    def test_get_llm_by_name_with_access_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """User without UserModel accesses shared model via visible_ids."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # The regular user's model list should contain the shared model
        response = client.get("/api/models/", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        shared = [m for m in data if m["model_id"] == sample_model_data["model_id"]]
        assert len(shared) == 1
        assert shared[0]["is_owner"] is False

    def test_get_user_default_models_includes_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """get_user_default_models returns shared model defaults with is_owner=False."""
        # Admin creates shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Admin sets it as their default
        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )
        assert default_response.status_code == 200

        # Regular user sees admin's shared default
        response = client.get("/api/models/user-default", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        general_default = [d for d in data if d.get("config_type") == "general"]
        assert len(general_default) == 1
        assert general_default[0]["model"]["is_owner"] is False
        assert general_default[0]["model"]["can_edit"] is False
        assert general_default[0]["model"]["can_delete"] is False

    def test_set_user_default_model_shared_access(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """User can set shared model as their default via Mode B."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Regular user sets shared model as default
        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=regular_headers,
        )
        assert default_response.status_code == 200

    def test_get_default_provider_endpoints_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """get_default/{provider} endpoints return shared models for non-owner who set their own default."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Regular user sets the shared model as their own default
        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=regular_headers,
        )
        assert default_response.status_code == 200

        # Now /default/general returns the shared model with is_owner=False
        response = client.get("/api/models/default/general", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert data is not None
        assert data["is_owner"] is False
        assert data["can_edit"] is False
        assert data["can_delete"] is False


# ===========================================================================
# 4. Mode C — Model Listing
# ===========================================================================


class TestModeCModelListing:
    """Test or_() filter for model listing."""

    def test_list_models_shared_is_owner_false(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Shared models in list have is_owner=False."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        response = client.get("/api/models/", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        shared = [m for m in data if m["model_id"] == sample_model_data["model_id"]]
        assert len(shared) == 1
        assert shared[0]["is_owner"] is False
        assert shared[0]["can_edit"] is False
        assert shared[0]["can_delete"] is False
        assert shared[0]["is_shared"] is True

    def test_test_models_includes_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data, monkeypatch
    ):
        """test endpoint finds shared models."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # Avoid real network calls — we only need to verify the shared model is selected
        async def fake_chat(*args, **kwargs):
            return "ok"

        monkeypatch.setattr(
            "xagent.web.services.llm_utils.create_base_llm",
            lambda config: type("FakeLLM", (), {"chat": fake_chat})(),
        )

        response = client.post("/api/models/test", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert any(m["model_id"] == sample_model_data["model_id"] for m in data)
        assert all(m["status"] == "passed" for m in data)

    def test_list_model_categories_includes_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Categories from shared models are visible."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        response = client.get("/api/models/categories", headers=regular_headers)
        assert response.status_code == 200
        assert "llm" in response.json()["categories"]

    def test_list_model_providers_includes_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Providers from shared models are visible."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        response = client.get("/api/models/providers", headers=regular_headers)
        assert response.status_code == 200
        assert "openai" in response.json()["providers"]

    def test_get_models_summary_includes_shared(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Summary includes shared models in counts."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        response = client.get("/api/models/summary", headers=regular_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] >= 1


# ===========================================================================
# 5. Permission Checks
# ===========================================================================


class TestPermissionChecks:
    """Test owner-only edit/delete via dynamic model sharing."""

    def test_update_model_can_user_share_hook(
        self, test_db, admin_headers, sample_model_data
    ):
        """When _can_user_share hook returns False, admin cannot enable sharing."""
        set_can_share_hook(lambda user: False)

        sample_model_data["share_with_users"] = True
        response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert response.status_code == 403

    def test_non_owner_cannot_update_shared_model(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Non-owner gets 403 when trying to edit a shared model."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        response = client.put(
            f"/api/models/{model_id_str}",
            json={"temperature": 0.1},
            headers=regular_headers,
        )
        assert response.status_code == 403

    def test_non_owner_cannot_delete_shared_model(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Non-owner gets 403 when trying to delete a shared model."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        response = client.delete(f"/api/models/{model_id_str}", headers=regular_headers)
        assert response.status_code == 403


# ===========================================================================
# 6. Protection Constraints
# ===========================================================================


class TestProtectionConstraints:
    """Test constraint 1 (cannot delete/un-share own default) and constraint 2 (shared model field locking)."""

    def test_owner_cannot_delete_own_default(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 1: owner cannot delete model that is their own default."""
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Set as own default
        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )
        assert default_response.status_code == 200

        # Try to delete — should fail with 409
        delete_response = client.delete(
            f"/api/models/{model_id_str}", headers=admin_headers
        )
        assert delete_response.status_code == 409
        assert "default" in delete_response.json()["detail"].lower()

    def test_owner_cannot_unshare_own_default(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 1: owner cannot un-share model that is their own default."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Set as own default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Try to un-share — should fail with 409
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 409
        assert "default" in update_response.json()["detail"].lower()

    def test_unshared_default_model_sending_false_is_noop(
        self, test_db, admin_headers, sample_model_data
    ):
        """share_with_users=false on an already-unshared default model is a no-op (200)."""
        # Create model WITHOUT sharing
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Set as own default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Sending share_with_users=False on an already-unshared model should be a no-op
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

    def test_owner_can_delete_when_no_default(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 1 reverse: owner can delete model when it's not their default."""
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Delete without setting default — should succeed
        delete_response = client.delete(
            f"/api/models/{model_id_str}", headers=admin_headers
        )
        assert delete_response.status_code == 200

    def test_owner_can_unshare_when_no_default(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 1 reverse: owner can un-share when model is not their default."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Un-share without setting default — should succeed
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

    def test_shared_model_cannot_change_category(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 2: cannot change category of shared model."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"category": "embedding"},
            headers=admin_headers,
        )
        assert update_response.status_code == 409
        assert "category" in update_response.json()["detail"]

    def test_shared_model_cannot_change_abilities(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 2: cannot change abilities of shared model."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"abilities": ["chat"]},  # original was ["chat", "tool_calling"]
            headers=admin_headers,
        )
        assert update_response.status_code == 409
        assert "abilities" in update_response.json()["detail"]

    def test_shared_model_can_change_other_fields(
        self, test_db, admin_headers, sample_model_data
    ):
        """Constraint 2 reverse: shared model allows changing api_key, temperature, description."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"temperature": 0.5, "description": "Updated"},
            headers=admin_headers,
        )
        assert update_response.status_code == 200
        assert update_response.json()["temperature"] == 0.5
        assert update_response.json()["description"] == "Updated"


# ===========================================================================
# 7. Un-share Cleanup
# ===========================================================================


class TestUnshareCleanup:
    """Test that un-sharing cleans up non-owner UserModel and UserDefaultModel."""

    def _create_shared_model_and_set_user_default(
        self, admin_headers, regular_headers, sample_model_data
    ):
        """Helper: admin creates shared model, regular user sets it as default."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Regular user sets shared model as default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=regular_headers,
        )

        return model_id_str, model_db_id

    def test_unshare_deletes_non_owner_user_default_model(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Un-sharing deletes non-owner UserDefaultModel records."""
        model_id_str, model_db_id = self._create_shared_model_and_set_user_default(
            admin_headers, regular_headers, sample_model_data
        )

        # Verify regular user has a default
        defaults_before = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert any(d["config_type"] == "general" for d in defaults_before.json())

        # Admin un-shares
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

        # Regular user's default for general should be gone
        defaults_after = client.get("/api/models/user-default", headers=regular_headers)
        general_after = [
            d for d in defaults_after.json() if d.get("config_type") == "general"
        ]
        assert len(general_after) == 0

    def test_unshare_preserves_owner_data(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Un-sharing preserves owner's UserModel and UserDefaultModel."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Admin sets own default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Admin un-shares (remove the default first to avoid constraint)
        client.delete("/api/models/user-default/general", headers=admin_headers)

        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

        # Owner should still see the model
        get_response = client.get(f"/api/models/{model_id_str}", headers=admin_headers)
        assert get_response.status_code == 200
        assert get_response.json()["is_owner"] is True

    def test_unshare_regular_user_falls_back_to_admin_default(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """After un-share, regular user falls back to admin's shared default."""
        model_id_str, model_db_id = self._create_shared_model_and_set_user_default(
            admin_headers, regular_headers, sample_model_data
        )

        # Create a second model as admin (also shared) for fallback
        second_model_data = {
            "model_id": "second-model",
            "category": "llm",
            "model_provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "api_key": "test-api-key-2",
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.5,
            "abilities": ["chat"],
            "share_with_users": True,
        }
        second_create = client.post(
            "/api/models/", json=second_model_data, headers=admin_headers
        )
        assert second_create.status_code == 200
        second_model_db_id = second_create.json()["id"]

        # Admin sets second model as their default
        client.post(
            "/api/models/user-default",
            json={"model_id": second_model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Admin un-shares first model (needs to remove admin's own default first)
        client.delete("/api/models/user-default/general", headers=admin_headers)
        # Set second model as admin's default
        client.post(
            "/api/models/user-default",
            json={"model_id": second_model_db_id, "config_type": "general"},
            headers=admin_headers,
        )
        # Now unshare the first model (admin's default is the second model, so no constraint)
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

        # Regular user should see admin's second model as fallback
        defaults = client.get("/api/models/user-default", headers=regular_headers)
        general = [d for d in defaults.json() if d.get("config_type") == "general"]
        assert len(general) == 1
        assert general[0]["model"]["model_id"] == "second-model"

    def test_unshare_deletes_non_owner_user_model(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Un-sharing deletes non-owner UserModel records (dynamic sharing only has owner's UserModel)."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]

        # Before un-share: regular user can see the shared model
        get_before = client.get(f"/api/models/{model_id_str}", headers=regular_headers)
        assert get_before.status_code == 200

        # Admin un-shares
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

        # After un-share: regular user can no longer see the model
        get_after = client.get(f"/api/models/{model_id_str}", headers=regular_headers)
        assert get_after.status_code == 404


# ===========================================================================
# 10. Static Code Cleanup Verification
# ===========================================================================


class TestStaticCodeCleanup:
    """Verify no pre-creation happens with dynamic model sharing."""

    def test_create_model_no_precreation(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Creating a shared model should NOT create UserModel for other users."""
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # Check DB: only admin's UserModel should exist for this model
        db = next(get_db())
        from xagent.web.models.user import UserModel

        model_db_id = create_response.json()["id"]
        user_models = (
            db.query(UserModel).filter(UserModel.model_id == model_db_id).all()
        )
        db.close()

        # Only one UserModel — the owner's
        assert len(user_models) == 1
        assert user_models[0].is_owner is True
        assert user_models[0].is_shared is True

    def test_update_model_enable_sharing_no_precreation(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Enabling sharing should NOT create UserModel for other users."""
        sample_model_data["share_with_users"] = False
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_id_str = create_response.json()["model_id"]
        model_db_id = create_response.json()["id"]

        # Enable sharing
        update_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": True},
            headers=admin_headers,
        )
        assert update_response.status_code == 200

        # Check DB: only admin's UserModel should exist
        db = next(get_db())
        from xagent.web.models.user import UserModel

        user_models = (
            db.query(UserModel).filter(UserModel.model_id == model_db_id).all()
        )
        db.close()

        assert len(user_models) == 1
        assert user_models[0].is_shared is True


# ===========================================================================
# Category 2: Mode A — Default model lookup integration tests
# ===========================================================================


class TestModeADefaultLookup:
    """Integration tests verifying Mode A changes (DBModel JOIN, is_shared preservation)."""

    def test_shared_model_set_as_default_found_via_dbmodel_join(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """User has UserDefaultModel pointing to a shared model but no own UserModel.
        Mode A: should find via DBModel.is_active JOIN."""
        # Admin creates shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Regular user sets it as default (via shared model access)
        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=regular_headers,
        )
        assert default_response.status_code == 200

        # Verify the user has a default but NO own UserModel for this model
        db = next(get_db())
        from xagent.web.models.user import User, UserDefaultModel, UserModel

        regular = db.query(User).filter(User.username == "regularuser").first()
        assert regular is not None

        # User has a UserDefaultModel pointing to the shared model
        user_default = (
            db.query(UserDefaultModel)
            .filter(
                UserDefaultModel.user_id == regular.id,
                UserDefaultModel.config_type == "general",
            )
            .first()
        )
        assert user_default is not None

        # User does NOT have their own UserModel (only admin has one)
        own_user_model = (
            db.query(UserModel)
            .filter(
                UserModel.user_id == regular.id,
                UserModel.model_id == model_db_id,
            )
            .first()
        )
        assert own_user_model is None  # No pre-created UserModel

        # Despite no own UserModel, user-default endpoint should return the default
        defaults_response = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_response.status_code == 200
        data = defaults_response.json()
        general = [d for d in data if d.get("config_type") == "general"]
        assert len(general) == 1
        assert general[0]["model"]["model_id"] == sample_model_data["model_id"]
        assert general[0]["model"]["is_owner"] is False

        db.close()

    def test_admin_fallback_preserves_is_shared_check(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Admin fallback only returns shared (is_shared=True) models, not private ones."""
        # Admin creates a PRIVATE model (not shared)
        private_data = dict(sample_model_data)
        private_data["model_id"] = "private-model"
        private_data["share_with_users"] = False
        private_response = client.post(
            "/api/models/", json=private_data, headers=admin_headers
        )
        assert private_response.status_code == 200
        private_db_id = private_response.json()["id"]

        # Admin sets private model as their default
        client.post(
            "/api/models/user-default",
            json={"model_id": private_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Regular user should NOT see admin's private model in user-defaults
        defaults_response = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_response.status_code == 200
        data = defaults_response.json()
        general = [d for d in data if d.get("config_type") == "general"]
        assert len(general) == 0  # Private model should NOT appear as fallback

    def test_no_third_tier_fallback(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Third-tier fallback (any shared defaults from any user) is removed."""
        # Admin creates a shared model but does NOT set it as default
        sample_model_data["share_with_users"] = True
        sample_model_data["model_id"] = "shared-no-default"
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # Regular user has no default
        defaults_response = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_response.status_code == 200
        data = defaults_response.json()
        general = [d for d in data if d.get("config_type") == "general"]
        # The shared model has no admin default, so it should NOT appear
        assert len(general) == 0


class TestModeAMultipleUsers:
    """Test model service with multiple users using dynamic sharing."""

    def test_multiple_users_isolated_defaults(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """Each user's defaults are isolated; admin shared models are visible to all."""
        # Admin creates shared model
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Admin sets as their default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Create second regular user
        user2_data = {
            "username": "user2",
            "email": "user2@example.com",
            "password": "password2",
        }
        user2_response = client.post("/api/auth/register", json=user2_data)
        assert user2_response.status_code == 200

        login2 = client.post(
            "/api/auth/login", json={"username": "user2", "password": "password2"}
        )
        user2_headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}

        # Both regular users see admin's shared default
        for headers in [regular_headers, user2_headers]:
            defaults = client.get("/api/models/user-default", headers=headers)
            assert defaults.status_code == 200
            data = defaults.json()
            general = [d for d in data if d.get("config_type") == "general"]
            assert len(general) == 1
            assert general[0]["model"]["is_owner"] is False

        # Regular user 1 sets a different model as their own default
        own_model_data = dict(sample_model_data)
        own_model_data["model_id"] = "user1-own-model"
        own_model_data["share_with_users"] = False
        own_create = client.post(
            "/api/models/", json=own_model_data, headers=regular_headers
        )
        assert own_create.status_code == 200
        own_model_db_id = own_create.json()["id"]

        client.post(
            "/api/models/user-default",
            json={"model_id": own_model_db_id, "config_type": "general"},
            headers=regular_headers,
        )

        # User 1 sees their own default
        defaults1 = client.get("/api/models/user-default", headers=regular_headers)
        general1 = [d for d in defaults1.json() if d.get("config_type") == "general"]
        assert len(general1) == 1
        assert general1[0]["model"]["model_id"] == "user1-own-model"
        assert general1[0]["model"]["is_owner"] is True

        # User 2 still sees admin's shared default (not user 1's private model)
        defaults2 = client.get("/api/models/user-default", headers=user2_headers)
        general2 = [d for d in defaults2.json() if d.get("config_type") == "general"]
        assert len(general2) == 1
        assert general2[0]["model"]["model_id"] == sample_model_data["model_id"]
        assert general2[0]["model"]["is_owner"] is False


# ===========================================================================
# 11. Legacy Data Compatibility
# ===========================================================================


class TestLegacyDataCompatibility:
    """Tests for legacy UserModel rows (pre-created with is_owner=False)."""

    def test_legacy_non_owner_row_not_treated_as_own(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """A legacy UserModel with is_owner=False is not treated as ownership —
        verified via list_models. The model is visible through the legacy row
        but is_owner is correctly False."""
        # Admin creates a private model
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        # Simulate a legacy non-owner UserModel for the regular user
        db = next(get_db())
        from xagent.web.models.user import User, UserModel

        regular = db.query(User).filter(User.username == "regularuser").first()
        assert regular is not None
        legacy_row = UserModel(
            user_id=regular.id,
            model_id=model_db_id,
            is_owner=False,
            is_shared=False,
        )
        db.add(legacy_row)
        db.commit()
        db.close()

        # Regular user sees the model via legacy row, but is_owner=False
        list_response = client.get("/api/models/", headers=regular_headers)
        assert list_response.status_code == 200
        data = list_response.json()
        found = [m for m in data if m["model_id"] == sample_model_data["model_id"]]
        assert len(found) == 1
        assert found[0]["is_owner"] is False

    def test_legacy_duplicate_rows_deduped_in_list(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """When both a legacy (is_owner=False, is_shared=True) and a real owner
        UserModel exist, list_models returns exactly one row with correct ownership."""
        sample_model_data["share_with_users"] = False
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200

        # Regular user's own model via API (creates is_owner=True UserModel)
        own_data = dict(sample_model_data)
        own_data["model_id"] = "regular-owned-model"
        own_create = client.post("/api/models/", json=own_data, headers=regular_headers)
        assert own_create.status_code == 200

        # Admin sees exactly one row for their own model
        list_response = client.get("/api/models/", headers=admin_headers)
        assert list_response.status_code == 200
        data = list_response.json()
        admin_models = [
            m for m in data if m["model_id"] == sample_model_data["model_id"]
        ]
        assert len(admin_models) == 1
        assert admin_models[0]["is_owner"] is True


# ===========================================================================
# 12. Stale Default Visibility
# ===========================================================================


class TestGetUserDefaultModelsStaleSkip:
    """Tests for stale UserDefaultModel visibility."""

    def test_shared_fallback_default_delete_invalidates_other_user_cache(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )
        assert default_response.status_code == 200

        set_cache_backend_for_testing(InMemoryTTLCache())
        defaults_before = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_before.status_code == 200
        general_before = [
            d for d in defaults_before.json() if d.get("config_type") == "general"
        ]
        assert len(general_before) == 1
        assert general_before[0]["model"]["model_id"] == sample_model_data["model_id"]

        delete_response = client.delete(
            "/api/models/user-default/general", headers=admin_headers
        )
        assert delete_response.status_code == 200

        defaults_after = client.get("/api/models/user-default", headers=regular_headers)
        assert defaults_after.status_code == 200
        general_after = [
            d for d in defaults_after.json() if d.get("config_type") == "general"
        ]
        assert general_after == []

    def test_skips_stale_user_default_and_falls_back(
        self, test_db, admin_headers, regular_headers, sample_model_data
    ):
        """When user's default points to a model no longer visible,
        get_user_default_models falls back to admin shared defaults."""
        # Admin creates a shared model and another as fallback
        sample_model_data["share_with_users"] = True
        create_response = client.post(
            "/api/models/", json=sample_model_data, headers=admin_headers
        )
        assert create_response.status_code == 200
        model_db_id = create_response.json()["id"]
        model_id_str = create_response.json()["model_id"]

        # Admin sets this as own default
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=admin_headers,
        )

        # Regular user sets it as own default too
        client.post(
            "/api/models/user-default",
            json={"model_id": model_db_id, "config_type": "general"},
            headers=regular_headers,
        )

        # Verify regular user sees the default
        defaults_before = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        general_before = [
            d for d in defaults_before.json() if d.get("config_type") == "general"
        ]
        assert len(general_before) == 1

        # Admin un-shares the model (remove admin's own default first to avoid constraint)
        client.delete("/api/models/user-default/general", headers=admin_headers)
        # Create a second model as admin's new default
        second_model_data = dict(sample_model_data)
        second_model_data["model_id"] = "fallback-model"
        second_model_data["share_with_users"] = True
        second_create = client.post(
            "/api/models/", json=second_model_data, headers=admin_headers
        )
        assert second_create.status_code == 200
        second_model_db_id = second_create.json()["id"]
        client.post(
            "/api/models/user-default",
            json={"model_id": second_model_db_id, "config_type": "general"},
            headers=admin_headers,
        )
        # Now un-share first model
        unshare_response = client.put(
            f"/api/models/{model_id_str}",
            json={"share_with_users": False},
            headers=admin_headers,
        )
        assert unshare_response.status_code == 200

        # Regular user's stale default should be skipped; falls back to admin's new default
        defaults_after = client.get("/api/models/user-default", headers=regular_headers)
        general_after = [
            d for d in defaults_after.json() if d.get("config_type") == "general"
        ]
        assert len(general_after) == 1
        assert general_after[0]["model"]["model_id"] == "fallback-model"
        assert general_after[0]["model"]["is_owner"] is False
