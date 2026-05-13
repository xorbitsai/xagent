"""Regression tests for task model-id handling in chat API."""

import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.auth import auth_router
from xagent.web.api.chat import AgentServiceManager, chat_router
from xagent.web.api.model import model_router
from xagent.web.models.database import Base, get_db, get_engine


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
test_app.include_router(chat_router)
test_app.dependency_overrides[get_db] = override_get_db

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
    from xagent.web.models.database import init_db

    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test.db")
    database_url = f"sqlite:///{temp_db_path}"

    init_db(db_url=database_url)

    engine = get_engine()
    yield

    Base.metadata.drop_all(bind=engine)
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="function")
def user1_headers(test_db):
    ensure_system_initialized()
    response = client.post(
        "/api/auth/register", json={"username": "user1", "password": "password123"}
    )
    assert response.status_code == 200

    login = client.post(
        "/api/auth/login", json={"username": "user1", "password": "password123"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def user2_headers(test_db):
    ensure_system_initialized()
    response = client.post(
        "/api/auth/register", json={"username": "user2", "password": "password123"}
    )
    assert response.status_code == 200

    login = client.post(
        "/api/auth/login", json={"username": "user2", "password": "password123"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def sample_model_data():
    return {
        "model_id": "user2-private-model",
        "category": "llm",
        "model_provider": "openai",
        "model_name": "gpt-4",
        "api_key": "test-api-key",
        "base_url": "https://api.openai.com/v1",
        "temperature": 0.7,
        "abilities": ["chat"],
        "description": "User2 private model",
        "share_with_users": False,
    }


def test_task_create_does_not_persist_inaccessible_model_ids(
    test_db, user1_headers, user2_headers, sample_model_data
):
    # User2 creates a private model.
    created = client.post("/api/models/", json=sample_model_data, headers=user2_headers)
    assert created.status_code == 200
    created_model = created.json()
    other_user_model_pk = str(created_model["id"])
    other_user_model_id = created_model["model_id"]

    # User1 tries to use User2's model by DB pk.
    resp = client.post(
        "/api/chat/task/create",
        json={
            "title": "test",
            "description": "desc",
            "llm_ids": [other_user_model_pk, None, None, None],
        },
        headers=user1_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] != other_user_model_id

    # User1 tries to use User2's model by internal stable model_id.
    resp2 = client.post(
        "/api/chat/task/create",
        json={
            "title": "test2",
            "description": "desc",
            "llm_ids": [other_user_model_id, None, None, None],
        },
        headers=user1_headers,
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["model_id"] != other_user_model_id


def test_standalone_task_create_defaults_to_think(test_db, user1_headers):
    resp = client.post(
        "/api/chat/task/create",
        json={"title": "test", "description": "desc"},
        headers=user1_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["execution_mode"] == "think"


def test_task_create_with_empty_delegate_selection_stores_none(test_db, user1_headers):
    from xagent.web.models.task import Task

    resp = client.post(
        "/api/chat/task/create",
        json={"title": "delegates", "description": "desc", "delegate_agent_ids": []},
        headers=user1_headers,
    )
    assert resp.status_code == 200

    db = next(get_db())
    try:
        task = db.query(Task).filter(Task.id == resp.json()["task_id"]).first()
        assert task is not None
        assert task.delegate_agent_ids is None
    finally:
        db.close()


def test_task_create_allows_shared_model_ids(
    test_db, user1_headers, user2_headers, sample_model_data
):
    """When admin shares a model, other users can use it in task creation (Mode B two-step)."""
    # Admin creates and shares a model
    admin_login = client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    )
    assert admin_login.status_code == 200
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

    shared_data = dict(sample_model_data)
    shared_data["share_with_users"] = True
    shared_data["model_id"] = "admin-shared-model"

    created = client.post("/api/models/", json=shared_data, headers=admin_headers)
    assert created.status_code == 200
    created_model = created.json()
    shared_model_id = created_model["model_id"]

    # User1 can use the shared model
    resp = client.post(
        "/api/chat/task/create",
        json={
            "title": "shared-model-task",
            "description": "desc",
            "llm_ids": [shared_model_id, None, None, None],
        },
        headers=user1_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # The task should use the shared model
    assert data["model_id"] == shared_model_id


def test_standalone_task_create_defaults_to_auto_in_v2(
    test_db, user1_headers, monkeypatch
):
    monkeypatch.setenv("XAGENT_AGENT_RUNTIME", "v2")

    resp = client.post(
        "/api/chat/task/create",
        json={"title": "test", "description": "desc"},
        headers=user1_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["execution_mode"] == "auto"


def test_get_task_llm_ids_preserves_stored_id_when_model_missing(test_db):
    ensure_system_initialized()
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        assert admin is not None
        task = Task(
            user_id=admin.id,
            title="t",
            description="d",
            status=TaskStatus.PENDING,
            model_id="deleted-model-id",
            small_fast_model_id="deleted-fast-id",
            visual_model_id="deleted-visual-id",
            compact_model_id="deleted-compact-id",
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        manager = AgentServiceManager()
        ids = manager._get_task_llm_ids(task, db)

        assert ids[0] == "deleted-model-id"
        assert ids[1] == "deleted-fast-id"
        assert ids[2] == "deleted-visual-id"
        assert ids[3] == "deleted-compact-id"
    finally:
        db.close()


def test_task_create_skips_stale_user_default(test_db, user1_headers):
    """When a user's default model is no longer visible, task falls back to admin shared."""
    from xagent.web.models.database import get_db
    from xagent.web.models.model import Model as DBModel
    from xagent.web.models.user import User, UserDefaultModel, UserModel

    db = next(get_db())
    try:
        user1 = db.query(User).filter(User.username == "user1").first()
        admin = db.query(User).filter(User.username == "admin").first()
        assert user1 is not None
        assert admin is not None

        # -- Admin shared fallback model --
        admin_shared_model = DBModel(
            model_id="admin-shared-fallback",
            category="llm",
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            abilities=["chat"],
            is_active=True,
        )
        db.add(admin_shared_model)
        db.commit()
        db.refresh(admin_shared_model)

        db.add(
            UserModel(
                user_id=admin.id,
                model_id=admin_shared_model.id,
                is_owner=True,
                is_shared=True,
            )
        )
        db.add(
            UserDefaultModel(
                user_id=admin.id,
                model_id=admin_shared_model.id,
                config_type="general",
            )
        )

        # -- User1's stale default (model with no UserModel row) --
        stale_model = DBModel(
            model_id="stale-inaccessible-model",
            category="llm",
            model_provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            abilities=["chat"],
            is_active=True,
        )
        db.add(stale_model)
        db.commit()
        db.refresh(stale_model)

        db.add(
            UserDefaultModel(
                user_id=user1.id,
                model_id=stale_model.id,
                config_type="general",
            )
        )
        db.commit()

        # Create task without specifying llm_ids — should resolve to admin shared fallback
        resp = client.post(
            "/api/chat/task/create",
            json={"title": "test-stale", "description": "desc"},
            headers=user1_headers,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Must NOT use the stale inaccessible model
        assert data.get("model_id") != "stale-inaccessible-model"
        # Must use admin's shared fallback model
        assert data.get("model_id") == "admin-shared-fallback"
    finally:
        db.close()


def test_task_create_rejects_draft_delegate_agents(test_db, user1_headers):
    from xagent.web.models.agent import Agent, AgentStatus
    from xagent.web.models.task import Task
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user1 = db.query(User).filter(User.username == "user1").first()
        assert user1 is not None

        published_agent = Agent(
            user_id=user1.id,
            name="published delegate",
            status=AgentStatus.PUBLISHED,
        )
        draft_agent = Agent(
            user_id=user1.id,
            name="draft delegate",
            status=AgentStatus.DRAFT,
        )
        db.add_all([published_agent, draft_agent])
        db.commit()
        db.refresh(published_agent)
        db.refresh(draft_agent)

        resp = client.post(
            "/api/chat/task/create",
            json={
                "title": "delegate-agent-task",
                "description": "desc",
                "delegate_agent_ids": [published_agent.id, draft_agent.id],
            },
            headers=user1_headers,
        )
        assert resp.status_code == 400
        assert "delegate agents" in resp.json()["detail"]

        created_task = (
            db.query(Task).filter(Task.title == "delegate-agent-task").first()
        )
        assert created_task is None
    finally:
        db.close()
