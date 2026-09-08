"""Test model management API functionality"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.model.model import ChatModelConfig, EmbeddingModelConfig
from xagent.web.api import model as model_module
from xagent.web.api.auth import auth_router
from xagent.web.api.model import model_router
from xagent.web.models.auto_model import AutoModelCandidate, AutoModelConfig
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import UserDefaultModel, UserModel
from xagent.web.services.llm_utils import (
    PLATFORM_MODEL_MANAGER,
    AutoModelUnavailableError,
    CoreStorage,
    PlatformModelIdentityError,
    PlatformModelStore,
)

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


def _platform_config(model_id: str, category: str = "llm"):
    common = {
        "id": model_id,
        "model_provider": "openai",
        "api_key": "platform-key",
        "base_url": "https://api.openai.com/v1",
    }
    if category == "embedding":
        return EmbeddingModelConfig(
            **common,
            model_name="text-embedding-3-small",
            abilities=["embedding"],
            dimension=1536,
        )
    return ChatModelConfig(**common, model_name="gpt-4", abilities=["chat"])


@pytest.mark.parametrize("path", ["/api/models/", "/api/models/register"])
def test_user_creation_rejects_platform_namespace(
    test_db, regular_headers, sample_model_data, path
):
    payload = {**sample_model_data, "model_id": "platform/forged"}

    response = client.post(path, headers=regular_headers, json=payload)

    assert response.status_code == 403
    db = next(get_db())
    try:
        assert db.query(DBModel).filter_by(model_id="platform/forged").first() is None
    finally:
        db.close()


@pytest.mark.parametrize("path", ["/api/models/", "/api/models/register"])
def test_user_creation_rejects_auto_router_namespace(
    test_db, regular_headers, sample_model_data, path
):
    payload = {**sample_model_data, "model_id": "auto-router-999"}

    response = client.post(path, headers=regular_headers, json=payload)

    assert response.status_code == 403
    assert "auto-router-" in response.json()["detail"]


def test_trusted_platform_store_persists_provenance_without_user_ownership(test_db):
    db = next(get_db())
    try:
        store = PlatformModelStore(db)
        created = store.create(_platform_config("platform/toby-embedding", "embedding"))

        assert created.managed_by == PLATFORM_MODEL_MANAGER
        assert store.get("platform/toby-embedding") is created
        assert db.query(UserModel).filter_by(model_id=created.id).first() is None
        assert db.query(UserDefaultModel).filter_by(model_id=created.id).first() is None
    finally:
        db.close()


def test_trusted_platform_store_does_not_adopt_preclaimed_collision(test_db):
    db = next(get_db())
    try:
        collision = DBModel(
            model_id="platform/preclaimed",
            category="llm",
            model_provider="openai",
            model_name="tenant-model",
            api_key="tenant-key",
            abilities=["chat"],
            is_active=True,
        )
        db.add(collision)
        db.commit()

        store = PlatformModelStore(db)
        assert store.get("platform/preclaimed") is None
        with pytest.raises(PlatformModelIdentityError, match="already claimed"):
            store.create(_platform_config("platform/preclaimed"))

        db.refresh(collision)
        assert collision.managed_by is None
        assert collision.model_name == "tenant-model"
        assert collision.is_active is True
    finally:
        db.close()


@pytest.mark.parametrize(
    ("initial_category", "requested_category"),
    [("llm", "embedding"), ("embedding", "llm")],
)
def test_platform_category_is_immutable_without_partial_mutation(
    test_db, admin_user, admin_headers, initial_category, requested_category
):
    db = next(get_db())
    model_id = f"platform/category-{initial_category}"
    try:
        created = PlatformModelStore(db).create(
            _platform_config(model_id, initial_category)
        )
        ownership = UserModel(
            user_id=admin_user["id"],
            model_id=created.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db.add(ownership)
        db.add(
            UserDefaultModel(
                user_id=admin_user["id"],
                model_id=created.id,
                config_type="general" if initial_category == "llm" else "embedding",
            )
        )
        db.commit()
        ownership_id = ownership.id
        default_id = db.query(UserDefaultModel).filter_by(model_id=created.id).one().id
    finally:
        db.close()

    response = client.put(
        f"/api/models/by-id/{quote(model_id, safe='')}",
        headers=admin_headers,
        json={
            "category": requested_category,
            "model_name": "mutated-name",
            "share_with_users": True,
        },
    )

    assert response.status_code == 409
    db = next(get_db())
    try:
        unchanged = db.query(DBModel).filter_by(model_id=model_id).one()
        assert unchanged.category == initial_category
        assert unchanged.model_name != "mutated-name"
        assert db.query(UserModel).filter_by(id=ownership_id).one().is_shared is False
        assert (
            db.query(UserDefaultModel).filter_by(id=default_id).one().model_id
            == unchanged.id
        )
    finally:
        db.close()


def test_preclaimed_platform_id_cannot_be_updated_deactivated_or_deleted(
    test_db, admin_user, admin_headers
):
    db = next(get_db())
    try:
        collision = DBModel(
            model_id="platform/tenant-row",
            category="llm",
            model_provider="openai",
            model_name="tenant-model",
            api_key="tenant-key",
            abilities=["chat"],
            is_active=True,
        )
        db.add(collision)
        db.flush()
        db.add(
            UserModel(
                user_id=admin_user["id"],
                model_id=collision.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_shared=False,
            )
        )
        db.commit()
        with pytest.raises(PlatformModelIdentityError):
            CoreStorage(db, DBModel).set_model_active(collision.model_id, False)
    finally:
        db.close()

    path = f"/api/models/by-id/{quote('platform/tenant-row', safe='')}"
    assert (
        client.put(
            path, headers=admin_headers, json={"model_name": "forged"}
        ).status_code
        == 403
    )
    assert client.delete(path, headers=admin_headers).status_code == 403

    db = next(get_db())
    try:
        unchanged = db.query(DBModel).filter_by(model_id="platform/tenant-row").one()
        assert unchanged.managed_by is None
        assert unchanged.model_name == "tenant-model"
        assert unchanged.is_active is True
    finally:
        db.close()


def test_ordinary_model_create_update_delete_remains_compatible(
    test_db, admin_headers, sample_model_data
):
    created = client.post("/api/models/", headers=admin_headers, json=sample_model_data)
    assert created.status_code == 200

    updated = client.put(
        "/api/models/test-openai-model",
        headers=admin_headers,
        json={"model_name": "gpt-4o"},
    )
    assert updated.status_code == 200
    assert updated.json()["model_name"] == "gpt-4o"

    deleted = client.delete("/api/models/test-openai-model", headers=admin_headers)
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_validate_provider_model_listing_honors_caller_supplied_timeout():
    """The listing helper must use the caller's budget, not a literal of its own.

    Guards against reintroducing the second hardcoded timeout this helper
    used to carry independently of ``test_model_connection``'s own budget
    (xorbitsai/xagent#1960): the two ``asyncio.wait_for`` layers wrapping a listing
    call (the endpoint's own, and this helper's inner one around the actual
    provider fetch) must always share the exact same number, sourced from
    the ``timeout_seconds`` parameter -- never a second literal in here.
    """

    async def slow_fetch(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return [{"id": "gpt-4o-mini", "abilities": ["chat"]}]

    with patch(
        "xagent.web.services.model_list_service.fetch_models_from_provider",
        side_effect=slow_fetch,
    ):
        with pytest.raises(asyncio.TimeoutError):
            await model_module._validate_provider_model_listing(
                provider="openai",
                model_name="gpt-4o-mini",
                api_key="key",
                base_url=None,
                timeout_seconds=0.05,
            )

        # A budget comfortably larger than the fetch's delay must succeed.
        # This is what actually pins the parameter as the value in effect:
        # a mutant that reverts to a hardcoded 10.0 inside the helper would
        # also pass this half alone, but would fail the 0.05s case above by
        # never timing out.
        await model_module._validate_provider_model_listing(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="key",
            base_url=None,
            timeout_seconds=1.0,
        )


class TestModelAPI:
    """Test model management API endpoints"""

    def test_auto_config_binds_existing_models_and_blocks_candidate_delete(
        self, test_db, regular_user, regular_headers, sample_model_data, monkeypatch
    ):
        first = client.post(
            "/api/models/",
            json={
                **sample_model_data,
                "abilities": ["chat", "tool_calling", "vision"],
            },
            headers=regular_headers,
        )
        second_payload = {
            **sample_model_data,
            "model_id": "test-second-model",
            "model_name": "gpt-4.1",
        }
        second = client.post(
            "/api/models/", json=second_payload, headers=regular_headers
        )
        assert first.status_code == 200
        assert second.status_code == 200

        class Catalog:
            @staticmethod
            def known_model_ids():
                return ("openai/gpt-5.5", "deepseek/deepseek-v4-flash")

            @staticmethod
            def get(profile_id):
                return SimpleNamespace(
                    input_modalities=("text", "image")
                    if profile_id == "openai/gpt-5.5"
                    else ("text",)
                )

        monkeypatch.setattr(
            "xagent.web.services.auto_model_service.load_router_profile_catalog",
            lambda: Catalog(),
        )
        with patch(
            "xagent.web.services.auto_model_service.load_router_profile_catalog",
            return_value=Catalog(),
        ):
            response = client.put(
                "/api/models/auto-config",
                headers=regular_headers,
                json={
                    "strategy": "quality",
                    "fallback_model_id": second.json()["id"],
                    "set_as_default": True,
                    "candidates": [
                        {
                            "target_model_id": first.json()["id"],
                            "routing_model_id": "openai/gpt-5.5",
                        },
                        {
                            "target_model_id": second.json()["id"],
                            "routing_model_id": "deepseek/deepseek-v4-flash",
                        },
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        # Legacy clients may still send a strategy, but configured Auto now always
        # uses xrouter's single-model "auto" policy.
        assert data["strategy"] == "balanced"
        assert data["auto_model"]["model_provider"] == "router"
        assert data["auto_model"]["model_name"] == "auto"
        assert data["auto_model"]["can_delete"] is False
        assert "vision" not in data["auto_model"]["abilities"]
        assert {candidate["routing_model_id"] for candidate in data["candidates"]} == {
            "openai/gpt-5.5",
            "deepseek/deepseek-v4-flash",
        }

        get_response = client.get("/api/models/auto-config", headers=regular_headers)
        assert get_response.status_code == 200
        assert get_response.json()["auto_model"]["id"] == data["auto_model"]["id"]
        assert get_response.json()["strategy"] == "balanced"

        defaults_response = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_response.status_code == 200
        general_default = next(
            item
            for item in defaults_response.json()
            if item["config_type"] == "general"
        )
        assert general_default["model_id"] == data["auto_model"]["id"]

        with patch(
            "xagent.web.services.auto_model_service.load_router_profile_catalog",
            return_value=Catalog(),
        ):
            update_response = client.put(
                "/api/models/auto-config",
                headers=regular_headers,
                json={
                    "fallback_model_id": second.json()["id"],
                    "candidates": [
                        {
                            "target_model_id": first.json()["id"],
                            "routing_model_id": "openai/gpt-5.5",
                        },
                        {
                            "target_model_id": second.json()["id"],
                            "routing_model_id": "deepseek/deepseek-v4-flash",
                        },
                    ],
                },
            )
        assert update_response.status_code == 200
        defaults_after_update = client.get(
            "/api/models/user-default", headers=regular_headers
        )
        assert defaults_after_update.status_code == 200
        assert any(
            item["config_type"] == "general"
            and item["model_id"] == data["auto_model"]["id"]
            for item in defaults_after_update.json()
        )

        list_response = client.get("/api/models/", headers=regular_headers)
        assert list_response.status_code == 200
        assert any(
            model["model_provider"] == "router" for model in list_response.json()
        )

        fake_llm = AsyncMock()
        fake_llm.chat.return_value = "ok"
        with patch.object(
            model_module.CoreStorage,
            "get_llm_by_id",
            return_value=fake_llm,
        ) as get_llm:
            all_test_response = client.post("/api/models/test", headers=regular_headers)
            auto_test_response = client.post(
                "/api/models/test",
                headers=regular_headers,
                json={"model_ids": [data["auto_model"]["model_id"]]},
            )
        assert all_test_response.status_code == 200
        assert auto_test_response.status_code == 200
        assert auto_test_response.json() == []
        assert data["auto_model"]["model_id"] not in {
            call.args[0] for call in get_llm.call_args_list
        }

        from xagent.core.model.chat.basic.router import RouterLLM
        from xagent.web.services.llm_utils import UserAwareModelStorage

        db = next(get_db())
        try:
            llm = UserAwareModelStorage(db).get_llm_by_id(
                data["auto_model"]["model_id"], regular_user["id"]
            )
            assert isinstance(llm, RouterLLM)
            assert llm.model_name == "auto"
            assert llm._candidate_models == (
                "openai/gpt-5.5",
                "deepseek/deepseek-v4-flash",
            )
            assert llm._fallback_model == "deepseek/deepseek-v4-flash"
            downstream = llm._downstream_resolver("openai/gpt-5.5")
            assert downstream.model_id == first.json()["model_id"]

            first_db_model = db.get(DBModel, first.json()["id"])
            second_db_model = db.get(DBModel, second.json()["id"])
            assert first_db_model is not None
            assert second_db_model is not None

            first_db_model.is_active = False
            db.flush()
            degraded_llm = UserAwareModelStorage(db).get_llm_by_id(
                data["auto_model"]["model_id"], regular_user["id"]
            )
            assert isinstance(degraded_llm, RouterLLM)
            assert degraded_llm._candidate_models == ("deepseek/deepseek-v4-flash",)
            assert degraded_llm._fallback_model == "deepseek/deepseek-v4-flash"

            second_db_model.is_active = False
            db.flush()
            with pytest.raises(
                AutoModelUnavailableError,
                match="Auto model has no active configured candidates",
            ):
                UserAwareModelStorage(db).get_llm_by_id(
                    data["auto_model"]["model_id"], regular_user["id"]
                )
            with (
                patch(
                    "xagent.web.services.llm_utils.create_llm_from_env"
                ) as env_fallback,
                pytest.raises(
                    AutoModelUnavailableError,
                    match="Auto model has no active configured candidates",
                ),
            ):
                UserAwareModelStorage(db).get_configured_defaults(regular_user["id"])
            env_fallback.assert_not_called()
        finally:
            db.rollback()
            db.close()

        delete_response = client.delete(
            f"/api/models/{first.json()['model_id']}", headers=regular_headers
        )
        assert delete_response.status_code == 409
        assert "Auto configuration" in delete_response.json()["detail"]

    @pytest.mark.parametrize("owner_action", ["unshare", "category", "delete"])
    def test_other_users_auto_binding_does_not_control_model_owner(
        self,
        test_db,
        regular_user,
        admin_user,
        admin_headers,
        owner_action,
    ):
        db = next(get_db())
        try:
            target = DBModel(
                model_id=f"cross-tenant-{owner_action}",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="owner-key",
                abilities=["chat"],
                is_active=True,
            )
            router_model = DBModel(
                model_id=f"auto-router-test-{owner_action}",
                category="llm",
                model_provider="router",
                model_name="auto",
                api_key="",
                abilities=["chat"],
                is_active=True,
            )
            db.add_all([target, router_model])
            db.flush()
            db.add(
                UserModel(
                    user_id=admin_user["id"],
                    model_id=target.id,
                    is_owner=True,
                    can_edit=True,
                    can_delete=True,
                    is_shared=owner_action == "unshare",
                )
            )
            config = AutoModelConfig(
                user_id=regular_user["id"],
                router_model_id=router_model.id,
                strategy="balanced",
                fallback_model_id=target.id,
            )
            db.add(config)
            db.flush()
            db.add(
                AutoModelCandidate(
                    config_id=config.id,
                    routing_model_id="openai/gpt-5.5",
                    target_model_id=target.id,
                )
            )
            db.commit()
            target_id = int(target.id)
            config_id = int(config.id)
        finally:
            db.close()

        if owner_action == "delete":
            response = client.delete(
                f"/api/models/cross-tenant-{owner_action}",
                headers=admin_headers,
            )
        else:
            update = (
                {"share_with_users": False}
                if owner_action == "unshare"
                else {"category": "embedding"}
            )
            response = client.put(
                f"/api/models/cross-tenant-{owner_action}",
                headers=admin_headers,
                json=update,
            )

        assert response.status_code == 200
        db = next(get_db())
        try:
            assert (
                db.query(AutoModelCandidate)
                .filter(AutoModelCandidate.target_model_id == target_id)
                .count()
                == 0
            )
            assert db.get(AutoModelConfig, config_id).fallback_model_id is None
        finally:
            db.close()

    def test_auto_config_rejects_duplicate_profile_mapping(
        self, test_db, regular_user, regular_headers, sample_model_data
    ):
        first = client.post(
            "/api/models/", json=sample_model_data, headers=regular_headers
        )
        second = client.post(
            "/api/models/",
            json={
                **sample_model_data,
                "model_id": "another-model",
                "model_name": "gpt-4.1",
            },
            headers=regular_headers,
        )
        response = client.put(
            "/api/models/auto-config",
            headers=regular_headers,
            json={
                "fallback_model_id": first.json()["id"],
                "candidates": [
                    {
                        "target_model_id": first.json()["id"],
                        "routing_model_id": "openai/gpt-5.5",
                    },
                    {
                        "target_model_id": second.json()["id"],
                        "routing_model_id": "openai/gpt-5.5",
                    },
                ],
            },
        )

        assert response.status_code == 422

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

    def test_test_connection_strips_whitespace_from_request_fields(
        self, test_db, regular_user, regular_headers
    ):
        """model_name/api_key/base_url padded with whitespace must reach the
        chat adapter trimmed — guards ModelConnectionTestRequest's
        strip_string_fields validator, which nothing else exercises."""
        captured = {}

        class FakeLLM:
            async def chat(self, messages, **kwargs):
                return {"content": "hi"}

        def fake_create_base_llm(config):
            captured["model_name"] = config.model_name
            captured["api_key"] = config.api_key
            captured["base_url"] = config.base_url
            return FakeLLM()

        with patch(
            "xagent.core.model.chat.basic.adapter.create_base_llm",
            side_effect=fake_create_base_llm,
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "openai",
                    "model_name": "  gpt-4o-mini  ",
                    "api_key": "  test-api-key  ",
                    "base_url": "  https://api.openai.com/v1  ",
                    "category": "llm",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        assert captured["model_name"] == "gpt-4o-mini"
        assert captured["api_key"] == "test-api-key"
        assert captured["base_url"] == "https://api.openai.com/v1"

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

    def test_test_connection_speech_elevenlabs_uses_listing_for_custom_asr(
        self, test_db, regular_user, regular_headers
    ):
        """ElevenLabs speech checks should not infer abilities from name prefixes."""
        with patch(
            "xagent.web.services.model_list_service.fetch_models_from_provider",
            new=AsyncMock(
                return_value=[
                    {
                        "id": "custom_transcriber",
                        "abilities": ["asr"],
                    }
                ]
            ),
        ) as mock_fetch:
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "elevenlabs",
                    "model_name": "custom_transcriber",
                    "api_key": "test-api-key",
                    "category": "speech",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        mock_fetch.assert_awaited_once()

    def test_test_connection_speech_elevenlabs_asr_uses_model_listing(
        self, test_db, regular_user, regular_headers
    ):
        """ElevenLabs ASR connection checks should not transcribe paid audio."""
        with patch(
            "xagent.web.services.model_list_service.fetch_models_from_provider",
            new=AsyncMock(
                return_value=[
                    {
                        "id": "scribe_v2",
                        "abilities": ["asr"],
                    }
                ]
            ),
        ) as mock_fetch:
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "elevenlabs",
                    "model_name": "scribe_v2",
                    "api_key": "test-api-key",
                    "category": "speech",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        mock_fetch.assert_awaited_once()

    def test_test_connection_sound_effect_does_not_require_catalog_match(
        self, test_db, regular_user, regular_headers
    ):
        """Sound-effect checks should probe auth without a billed generation."""
        sound_effect_model = Mock()
        sound_effect_model.validate_connection = AsyncMock(return_value=None)
        sound_effect_model.aclose = AsyncMock(return_value=None)

        with patch(
            "xagent.core.model.sound_effect.create_sound_effect_model",
            return_value=sound_effect_model,
        ) as create_model:
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "elevenlabs",
                    "model_name": "future-sfx-model",
                    "api_key": "test-api-key",
                    "category": "sound_effect",
                    "abilities": ["generate"],
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        assert response.json()["status"] == "passed"
        assert create_model.call_args.args[0].model_name == "future-sfx-model"
        sound_effect_model.validate_connection.assert_awaited_once_with()
        sound_effect_model.aclose.assert_awaited_once_with()

    def test_test_connection_llm_timeout_reports_app_budget_not_network(
        self, test_db, regular_user, regular_headers
    ):
        """A connection-test timeout must name the app's own wait budget.

        xorbitsai/xagent#1960: the old message ("Please check your network connection
        and provider status") told the user to go check their network when
        the actual cause was this endpoint's own wait budget (10 seconds at
        the time) expiring before a slow or reasoning-heavy model answered.
        The provider was never shown to be unhealthy.
        """

        class SlowLLM:
            async def chat(self, messages, **kwargs):
                raise asyncio.TimeoutError()

        with patch(
            "xagent.core.model.chat.basic.adapter.create_base_llm",
            return_value=SlowLLM(),
        ):
            response = client.post(
                "/api/models/test-connection",
                json={
                    "model_provider": "openai",
                    "model_name": "gpt-4o-mini",
                    "api_key": "test-api-key",
                    "base_url": "https://api.openai.com/v1",
                    "category": "llm",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["message"] == "Connection timed out"
        # Pin the contract value independently of the production constant:
        # deriving the expected copy from _CONNECTION_TEST_TIMEOUT_SECONDS
        # would keep this test green if the budget silently regressed.
        assert model_module._CONNECTION_TEST_TIMEOUT_SECONDS == 60.0
        assert "60 seconds" in data["error"]
        assert "network" not in data["error"].lower()

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

    def test_create_sound_effect_model_and_set_default(
        self,
        test_db,
        regular_user,
        regular_headers,
    ):
        create_response = client.post(
            "/api/models/",
            json={
                "model_id": "sound-effect-default",
                "category": "sound_effect",
                "model_provider": "elevenlabs",
                "model_name": "eleven_text_to_sound_v2",
                "api_key": "test-key",
                "abilities": ["generate"],
            },
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        assert create_response.json()["category"] == "sound_effect"

        default_response = client.post(
            "/api/models/user-default",
            json={
                "model_id": create_response.json()["id"],
                "config_type": "sound_effect",
            },
            headers=regular_headers,
        )
        assert default_response.status_code == 200
        assert default_response.json()["config_type"] == "sound_effect"

    def test_create_music_model_and_set_default(
        self,
        test_db,
        regular_user,
        regular_headers,
    ):
        create_response = client.post(
            "/api/models/",
            json={
                "model_id": "music-default",
                "category": "music",
                "model_provider": "elevenlabs",
                "model_name": "music_v2",
                "api_key": "test-key",
                "abilities": ["generate"],
            },
            headers=regular_headers,
        )
        assert create_response.status_code == 200
        assert create_response.json()["category"] == "music"

        default_response = client.post(
            "/api/models/user-default",
            json={
                "model_id": create_response.json()["id"],
                "config_type": "music",
            },
            headers=regular_headers,
        )
        assert default_response.status_code == 200
        assert default_response.json()["config_type"] == "music"

    def test_transcribe_speech_requires_asr_model(
        self,
        test_db,
        regular_user,
        regular_headers,
    ):
        response = client.post(
            "/api/models/speech/transcribe",
            files={"file": ("voice.webm", b"audio-bytes", "audio/webm")},
            headers=regular_headers,
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "No ASR model is configured"

    def test_transcribe_speech_rejects_oversized_audio(
        self,
        test_db,
        regular_user,
        regular_headers,
    ):
        with patch("xagent.web.api.model.MAX_TRANSCRIBE_UPLOAD_BYTES", 4):
            response = client.post(
                "/api/models/speech/transcribe",
                files={"file": ("voice.webm", b"audio", "audio/webm")},
                headers=regular_headers,
            )

        assert response.status_code == 413
        assert response.json()["detail"] == "Audio file is too large"

    def test_transcribe_speech_uses_user_default_asr_model(
        self,
        test_db,
        regular_user,
        regular_headers,
        sample_speech_model_data,
    ):
        first_model = dict(sample_speech_model_data)
        first_model["model_id"] = "test-asr-first"
        first_model["model_name"] = "asr-first"
        first_model["abilities"] = ["asr"]
        second_model = dict(sample_speech_model_data)
        second_model["model_id"] = "test-asr-second"
        second_model["model_name"] = "asr-second"
        second_model["abilities"] = ["asr"]

        first_response = client.post(
            "/api/models/",
            json=first_model,
            headers=regular_headers,
        )
        assert first_response.status_code == 200
        second_response = client.post(
            "/api/models/",
            json=second_model,
            headers=regular_headers,
        )
        assert second_response.status_code == 200

        default_response = client.post(
            "/api/models/user-default",
            json={"model_id": second_response.json()["id"], "config_type": "asr"},
            headers=regular_headers,
        )
        assert default_response.status_code == 200

        class FakeASR:
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.calls = []
                self.closed = False

            async def transcribe(self, audio, language=None, format=None):
                self.calls.append(
                    {"audio": audio, "language": language, "format": format}
                )
                return f"text from {self.model_name}"

            async def aclose(self):
                self.closed = True

        created_models = []

        def create_fake_asr(db_model):
            fake_asr = FakeASR(db_model.model_name)
            created_models.append(fake_asr)
            return fake_asr

        with patch(
            "xagent.core.model.asr.adapter.get_asr_model_instance",
            side_effect=create_fake_asr,
        ):
            response = client.post(
                "/api/models/speech/transcribe",
                files={"file": ("voice.webm", b"audio-bytes", "audio/webm")},
                data={"language": "en"},
                headers=regular_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "text from asr-second"
        assert data["model_id"] == "test-asr-second"
        assert len(created_models) == 1
        assert created_models[0].calls == [
            {"audio": b"audio-bytes", "language": "en", "format": "webm"}
        ]
        assert created_models[0].closed is True

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
        assert deepseek["category"] == ["llm"]
        assert deepseek["default_base_url"] == "https://api.deepseek.com"

    def test_list_supported_providers_includes_multi_category_provider(
        self, test_db, regular_user, regular_headers
    ):
        response = client.get(
            "/api/models/providers/supported",
            headers=regular_headers,
        )

        assert response.status_code == 200
        providers = response.json()["providers"]
        openai = next(
            (provider for provider in providers if provider["id"] == "openai"),
            None,
        )
        assert openai is not None
        assert openai["name"] == "OpenAI"
        assert openai["category"] == ["llm", "embedding"]
        assert openai["default_base_url"] == "https://api.openai.com/v1"

    def test_list_supported_providers_includes_openai_compatible(
        self, test_db, regular_user, regular_headers
    ):
        response = client.get(
            "/api/models/providers/supported",
            headers=regular_headers,
        )

        assert response.status_code == 200
        providers = response.json()["providers"]
        openai_compatible = next(
            (
                provider
                for provider in providers
                if provider["id"] == "openai-compatible"
            ),
            None,
        )
        assert openai_compatible is not None
        assert openai_compatible["name"] == "OpenAI-Compatible"
        assert openai_compatible["category"] == ["llm", "embedding"]
        assert openai_compatible["requires_base_url"] is True
        assert openai_compatible.get("default_base_url") is None

    def test_fetch_openai_compatible_provider_models_is_wired(
        self, test_db, regular_user, regular_headers, monkeypatch
    ):
        """Regression test: openai-compatible must be registered in
        PROVIDER_FETCHERS, or this endpoint 400s with "Unsupported provider".
        Only the network-facing SDK call is faked; the real PROVIDER_FETCHERS
        entry (or absence of one) is exercised as-is."""
        from xagent.core.model.chat.basic.openai import OpenAILLM

        async def fake_list_available_models(api_key, base_url=None):
            return [
                {
                    "id": "custom-model",
                    "object": "model",
                    "owned_by": "openai-compatible",
                }
            ]

        monkeypatch.setattr(
            OpenAILLM, "list_available_models", fake_list_available_models
        )

        response = client.post(
            "/api/models/providers/openai-compatible/models",
            json={
                "api_key": "test-api-key",
                "base_url": "https://custom.example.com/v1",
            },
            headers=regular_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert [model["id"] for model in data["models"]] == ["custom-model"]

    def test_fetch_provider_models_strips_whitespace_from_api_key_and_base_url(
        self, test_db, regular_user, regular_headers, monkeypatch
    ):
        """A base_url/api_key padded with whitespace must reach the fetcher
        trimmed, not verbatim — otherwise a value like " https://foo.com "
        would silently reach the downstream provider with stray whitespace."""
        from xagent.core.model.chat.basic.openai import OpenAILLM

        captured = {}

        async def fake_list_available_models(api_key, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            return []

        monkeypatch.setattr(
            OpenAILLM, "list_available_models", fake_list_available_models
        )

        response = client.post(
            "/api/models/providers/openai-compatible/models",
            json={
                "api_key": "  test-api-key  ",
                "base_url": "  https://custom.example.com/v1  ",
            },
            headers=regular_headers,
        )

        assert response.status_code == 200
        assert captured["api_key"] == "test-api-key"
        assert captured["base_url"] == "https://custom.example.com/v1"

    def test_fetch_provider_models_requires_base_url_for_openai_compatible(
        self, test_db, regular_user, regular_headers
    ):
        """openai-compatible is marked requires_base_url in provider metadata;
        omitting base_url must be rejected server-side rather than silently
        falling back to the OpenAI SDK's default endpoint."""
        response = client.post(
            "/api/models/providers/openai-compatible/models",
            json={"api_key": "test-api-key"},
            headers=regular_headers,
        )

        assert response.status_code == 400
        assert "base_url is required" in response.json()["detail"]

    def test_fetch_provider_models_requires_base_url_for_xinference(
        self, test_db, regular_user, regular_headers
    ):
        response = client.post(
            "/api/models/providers/xinference/models",
            json={"api_key": "test-api-key"},
            headers=regular_headers,
        )

        assert response.status_code == 400
        assert "base_url is required" in response.json()["detail"]

    def test_list_supported_providers_includes_elevenlabs_audio_generation(
        self, test_db, regular_user, regular_headers
    ):
        response = client.get(
            "/api/models/providers/supported",
            headers=regular_headers,
        )

        assert response.status_code == 200
        providers = response.json()["providers"]
        elevenlabs = next(
            (provider for provider in providers if provider["id"] == "elevenlabs"),
            None,
        )
        assert elevenlabs is not None
        assert elevenlabs["name"] == "ElevenLabs"
        assert elevenlabs["category"] == ["speech", "sound_effect", "music"]

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
        assert volcengine["category"] == ["video"]
        assert (
            volcengine["default_base_url"] == "https://ark.cn-beijing.volces.com/api/v3"
        )
        assert byteplus is not None
        assert byteplus["name"] == "BytePlus Ark"
        assert byteplus["category"] == ["video"]
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

    def test_fetch_dashscope_embedding_models_uses_curated_list(
        self, test_db, regular_user, regular_headers
    ):
        response = client.post(
            "/api/models/providers/dashscope/models",
            json={"api_key": "test-api-key", "category": "embedding"},
            headers=regular_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert [model["id"] for model in data["models"]] == [
            "text-embedding-v4",
            "text-embedding-v3",
        ]
        assert all(model["owned_by"] == "dashscope" for model in data["models"])

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

    @pytest.mark.parametrize(
        ("model_name", "temperature", "expected_default_temperature"),
        [
            ("openai/gpt-5.6-sol", None, None),
            ("gpt-4o", 0.0, 0.0),
            ("o1", 0.5, None),
            ("gpt-4o", 0.5, 0.5),
        ],
        ids=[
            "unset_temperature_is_never_invented",
            "explicit_zero_is_not_swallowed",
            "reasoning_model_still_rejects_explicit_temperature",
            "non_reasoning_model_forwards_explicit_temperature",
        ],
    )
    def test_test_connection_llm_default_temperature_matches_request(
        self,
        test_db,
        regular_user,
        regular_headers,
        model_name,
        temperature,
        expected_default_temperature,
    ):
        """The connection test must forward only a temperature the caller
        actually set, never invent one, and never let 0.0 be treated as
        falsy."""
        captured = {}

        def fake_create_base_llm(config):
            captured["default_temperature"] = config.default_temperature

            class FakeLLM:
                async def chat(self, messages, **kwargs):
                    return {"content": "hi"}

            return FakeLLM()

        payload = {
            "model_provider": "openai",
            "model_name": model_name,
            "api_key": "test-api-key",
            "base_url": "https://api.openai.com/v1",
            "category": "llm",
        }
        if temperature is not None:
            payload["temperature"] = temperature

        with patch(
            "xagent.core.model.chat.basic.adapter.create_base_llm",
            side_effect=fake_create_base_llm,
        ):
            response = client.post(
                "/api/models/test-connection",
                json=payload,
                headers=regular_headers,
            )

        assert response.status_code == 200
        assert captured["default_temperature"] == expected_default_temperature

    @pytest.mark.parametrize(
        ("model_name", "expected_chat_kwargs"),
        [
            ("o1", {}),
            ("gpt-4o", {"max_tokens": 16}),
        ],
        ids=[
            "reasoning_model_lets_adapter_pick_max_tokens",
            "non_reasoning_model_still_caps_max_tokens_at_16",
        ],
    )
    def test_test_connection_llm_max_tokens_matches_reasoning_heuristic(
        self,
        test_db,
        regular_user,
        regular_headers,
        model_name,
        expected_chat_kwargs,
    ):
        """max_tokens handling is unrelated to the temperature fix and must
        stay pinned to the existing reasoning-model heuristic."""
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
                    "model_provider": "openai",
                    "model_name": model_name,
                    "api_key": "test-api-key",
                    "base_url": "https://api.openai.com/v1",
                    "category": "llm",
                },
                headers=regular_headers,
            )

        assert response.status_code == 200
        mock_llm.chat.assert_awaited_once_with(
            [{"role": "user", "content": "Hello"}], **expected_chat_kwargs
        )
