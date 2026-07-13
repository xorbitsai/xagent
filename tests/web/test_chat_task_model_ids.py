"""Regression tests for task model-id handling in chat API."""

import logging
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.web.api.auth import auth_router
from xagent.web.api.chat import (
    AgentServiceManager,
    _load_agent_for_task_runtime,
    chat_router,
)
from xagent.web.api.model import model_router
from xagent.web.models.database import Base, get_db, get_engine
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy


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


@pytest.fixture(autouse=True)
def reset_workforce_policy():
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


@pytest.fixture(scope="function")
def user1_headers(test_db):
    ensure_system_initialized()
    response = client.post(
        "/api/auth/register",
        json={
            "username": "user1",
            "email": "user1@example.com",
            "password": "password123",
        },
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
        "/api/auth/register",
        json={
            "username": "user2",
            "email": "user2@example.com",
            "password": "password123",
        },
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


class DummyLLM(BaseLLM):
    def __init__(self, name: str):
        self._name = name

    @property
    def abilities(self):
        return ["chat"]

    @property
    def model_name(self):
        return self._name

    @property
    def supports_thinking_mode(self):
        return False

    async def chat(self, messages, **kwargs):
        return "ok"


def test_agent_builder_llm_overlay_preserves_resolved_task_llms():
    manager = AgentServiceManager()
    task_llm = DummyLLM("task-qwen")
    task_compact = DummyLLM("task-compact")

    merged = manager._merge_agent_builder_llms(
        (task_llm, None, task_llm, task_compact),
        (None, None, None, None),
    )

    assert merged == (task_llm, None, task_llm, task_compact)


def test_agent_builder_llm_overlay_uses_accessible_agent_llms():
    manager = AgentServiceManager()
    task_llm = DummyLLM("task-qwen")
    agent_llm = DummyLLM("agent-model")

    merged = manager._merge_agent_builder_llms(
        (task_llm, None, task_llm, None),
        (agent_llm, None, None, agent_llm),
    )

    assert merged == (agent_llm, None, task_llm, agent_llm)


def test_runtime_config_preserves_task_llm_when_agent_model_is_unavailable():
    manager = AgentServiceManager()
    task_llm = DummyLLM("task-qwen")
    task = MagicMock(
        agent_type="assistant",
        model_name="qwen3.6-plus",
        compact_model_name=None,
        execution_mode="balanced",
        agent_id=9,
        user_id=71,
    )
    user = MagicMock(id=71)
    from xagent.web.models.agent import AgentStatus

    agent = MagicMock(
        id=9,
        user_id=71,
        name="Published Agent",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = agent

    fake_agent_builder_config = {
        "llms": (None, None, None, None),
        "saved_model_ids": {"general": 123},
        "saved_model_descriptors": {
            "general": {
                "pk": 123,
                "model_id": "glm4.6v",
                "model_name": "glm4.6v",
            }
        },
        "execution_mode": "balanced",
        "instructions": "",
        "skills": [],
        "knowledge_bases": [],
        "tool_categories": [],
    }

    # LLM resolution + agent-builder config loading moved to module-
    # level ``resolve_task_runtime_config_core`` (called by both the
    # instance method here and ``load_task_setup_snapshot_sync``);
    # patch the helpers it actually invokes.
    with (
        patch(
            "xagent.web.services.llm_utils.load_agent_builder_config",
            return_value=fake_agent_builder_config,
        ) as load_cfg_mock,
        patch(
            "xagent.web.services.llm_utils.UserAwareModelStorage.resolve_llms_from_names",
            return_value=(task_llm, None, None, None),
        ),
        patch(
            "xagent.web.services.llm_utils.make_normalize_model_id",
            return_value=lambda mid, mname: mname,
        ),
    ):
        runtime_config = manager._resolve_task_runtime_config(
            task_id=42,
            task=task,
            db=db,
            user=user,
        )

    assert runtime_config["task_llm"] is task_llm
    load_cfg_mock.assert_called_once_with(agent, db, 71)


def test_runtime_config_uses_accessible_agent_model_over_task_baseline():
    manager = AgentServiceManager()
    task_llm = DummyLLM("task-qwen")
    agent_llm = DummyLLM("agent-qwen")
    task = MagicMock(
        agent_type="assistant",
        model_name="qwen3.6-plus",
        compact_model_name=None,
        execution_mode="balanced",
        agent_id=9,
        user_id=71,
    )
    user = MagicMock(id=71)
    db = MagicMock()
    from xagent.web.models.agent import AgentStatus

    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        id=9,
        user_id=71,
        name="Published Agent",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
    )

    fake_agent_builder_config = {
        "llms": (agent_llm, None, None, None),
        "saved_model_ids": {"general": 123},
        "saved_model_descriptors": {},
        "execution_mode": "balanced",
        "instructions": "",
        "skills": [],
        "knowledge_bases": [],
        "tool_categories": [],
    }

    with (
        patch(
            "xagent.web.services.llm_utils.load_agent_builder_config",
            return_value=fake_agent_builder_config,
        ),
        patch(
            "xagent.web.services.llm_utils.UserAwareModelStorage.resolve_llms_from_names",
            return_value=(task_llm, None, None, None),
        ),
        patch(
            "xagent.web.services.llm_utils.make_normalize_model_id",
            return_value=lambda mid, mname: mname,
        ),
    ):
        runtime_config = manager._resolve_task_runtime_config(
            task_id=42,
            task=task,
            db=db,
            user=user,
        )

    assert runtime_config["task_llm"] is agent_llm


def test_pick_default_llm_warns_with_agent_context_when_builder_config_present(caplog):
    """Agent builder fallback should log human-readable model identifiers."""
    default_llm = DummyLLM("default-llm")

    with caplog.at_level(logging.WARNING, logger="xagent.web.api.chat"):
        chosen = AgentServiceManager._pick_default_llm_with_warning(
            default_llm,
            task_id=42,
            has_agent_builder_config=True,
            agent_id=7,
            saved_model_ids={"general": 11, "small_fast": None},
            saved_model_descriptors={
                "general": {
                    "pk": 11,
                    "model_id": "gpt-4o",
                    "model_name": "gpt-4o-2024-08-06",
                },
            },
            user_id=99,
        )

    assert chosen is default_llm
    matched = [
        rec
        for rec in caplog.records
        if "falling back to default LLM" in rec.getMessage()
    ]
    assert len(matched) == 1
    message = matched[0].getMessage()
    assert "task_id=42" in message
    assert "agent_id=7" in message
    assert "user_id=99" in message
    # Descriptor fields are preferred over raw DB pks.
    assert "gpt-4o" in message
    assert "gpt-4o-2024-08-06" in message
    assert "agent_saved_models=" in message
    assert "fallback_model=default-llm" in message


def test_pick_default_llm_falls_back_to_saved_model_ids_when_descriptors_missing(
    caplog,
):
    """Without descriptors the warning falls back to raw saved_model_ids."""
    default_llm = DummyLLM("default-llm")

    with caplog.at_level(logging.WARNING, logger="xagent.web.api.chat"):
        AgentServiceManager._pick_default_llm_with_warning(
            default_llm,
            task_id=1,
            has_agent_builder_config=True,
            agent_id=2,
            saved_model_ids={"general": 11},
            user_id=3,
        )

    matched = [
        rec
        for rec in caplog.records
        if "falling back to default LLM" in rec.getMessage()
    ]
    assert len(matched) == 1
    message = matched[0].getMessage()
    assert "agent_saved_models=" in message
    assert "general" in message
    assert "11" in message


def test_pick_default_llm_warns_with_task_only_message_when_no_builder_config(caplog):
    """Without agent builder config we still warn but skip agent-specific fields."""
    default_llm = DummyLLM("default-llm")

    with caplog.at_level(logging.WARNING, logger="xagent.web.api.chat"):
        chosen = AgentServiceManager._pick_default_llm_with_warning(
            default_llm,
            task_id=42,
            has_agent_builder_config=False,
            agent_id=None,
            saved_model_ids=None,
            user_id=None,
        )

    assert chosen is default_llm
    matched = [
        rec
        for rec in caplog.records
        if "no valid LLM configuration" in rec.getMessage()
    ]
    assert len(matched) == 1
    assert "Task 42" in matched[0].getMessage()
    assert "default-llm" in matched[0].getMessage()


def test_pick_default_llm_raises_when_no_default_available():
    """If neither task/agent nor global default resolves, fail with a clear error."""
    with pytest.raises(HTTPException) as exc_info:
        AgentServiceManager._pick_default_llm_with_warning(
            None,
            task_id=42,
            has_agent_builder_config=True,
            agent_id=7,
            saved_model_ids={"general": 11},
            user_id=99,
        )

    assert exc_info.value.status_code == 500
    assert "no global default model" in str(exc_info.value.detail)


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


def test_standalone_task_create_defaults_to_auto(test_db, user1_headers):
    resp = client.post(
        "/api/chat/task/create",
        json={"title": "test", "description": "desc"},
        headers=user1_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["execution_mode"] == "auto"


def test_web_task_detail_cache_reuses_response_until_task_changes(
    test_db, user1_headers, monkeypatch
):
    from xagent.web.services.hot_path_cache import (
        InMemoryTTLCache,
        set_cache_backend_for_testing,
    )

    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        create_resp = client.post(
            "/api/chat/task/create",
            json={"title": "cache-test", "description": "desc"},
            headers=user1_headers,
        )
        assert create_resp.status_code == 200
        task_id = create_resp.json()["task_id"]

        first = client.get(f"/api/chat/task/{task_id}", headers=user1_headers)
        assert first.status_code == 200
        assert first.json()["title"] == "cache-test"

        def fail_if_uncached(*args, **kwargs):
            raise AssertionError("cache miss reached model id resolution")

        monkeypatch.setattr(
            AgentServiceManager,
            "_get_task_llm_ids",
            fail_if_uncached,
        )
        cached = client.get(f"/api/chat/task/{task_id}", headers=user1_headers)
        assert cached.status_code == 200
        assert cached.json()["title"] == "cache-test"

        monkeypatch.undo()
        update = client.put(
            f"/api/chat/task/{task_id}",
            json={"title": "cache-test-updated"},
            headers=user1_headers,
        )
        assert update.status_code == 200

        refreshed = client.get(f"/api/chat/task/{task_id}", headers=user1_headers)
        assert refreshed.status_code == 200
        assert refreshed.json()["title"] == "cache-test-updated"
    finally:
        set_cache_backend_for_testing(None)


def test_get_task_returns_token_usage_grouped_by_actual_model(test_db, user1_headers):
    from xagent.web.models.task import Task

    create_resp = client.post(
        "/api/chat/task/create",
        json={"title": "model-usage-test", "description": "desc"},
        headers=user1_headers,
    )
    assert create_resp.status_code == 200
    task_id = create_resp.json()["task_id"]

    db = next(get_db())
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.input_tokens = 120
        task.output_tokens = 30
        task.total_tokens = 150
        task.token_usage_details = [
            {
                "type": "input",
                "tokens": 100,
                "model": "gpt-4.1",
                "model_id": "main",
            },
            {
                "type": "output",
                "tokens": 25,
                "model": "gpt-4.1",
                "model_id": "main",
            },
            {
                "type": "input",
                "tokens": 20,
                "model": "gpt-4.1-mini",
                "model_id": "compact",
            },
            {
                "type": "output",
                "tokens": 5,
                "model": "gpt-4.1-mini",
                "model_id": "compact",
            },
        ]
        db.commit()
    finally:
        db.close()

    response = client.get(f"/api/chat/task/{task_id}", headers=user1_headers)
    assert response.status_code == 200
    assert response.json()["model_usage"] == [
        {
            "model_id": "main",
            "model_name": "gpt-4.1",
            "input_tokens": 100,
            "output_tokens": 25,
        },
        {
            "model_id": "compact",
            "model_name": "gpt-4.1-mini",
            "input_tokens": 20,
            "output_tokens": 5,
        },
    ]


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


def test_get_tasks_hides_invisible_tasks_by_default(test_db, user1_headers):
    from xagent.web.models.database import get_db
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == "user1").one()
        normal_task = Task(
            user_id=user.id,
            title="normal task",
            description="normal",
            status=TaskStatus.PENDING,
        )
        sdk_task = Task(
            user_id=user.id,
            title="sdk task",
            description="sdk",
            status=TaskStatus.PENDING,
            source="sdk",
            is_visible=False,
        )
        db.add_all([normal_task, sdk_task])
        db.commit()

        default_response = client.get("/api/chat/tasks", headers=user1_headers)
        assert default_response.status_code == 200
        assert [task["title"] for task in default_response.json()["tasks"]] == [
            "normal task"
        ]

        include_response = client.get(
            "/api/chat/tasks?include_hidden=true",
            headers=user1_headers,
        )
        assert include_response.status_code == 200
        assert {task["title"] for task in include_response.json()["tasks"]} == {
            "normal task",
            "sdk task",
        }
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


def test_task_create_rejects_agent_id_from_another_user(
    test_db, user1_headers, user2_headers
):
    from xagent.web.models.agent import Agent, AgentStatus
    from xagent.web.models.database import get_db
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user2 = db.query(User).filter(User.username == "user2").first()
        assert user2 is not None

        agent = Agent(
            user_id=user2.id,
            name="user2-private-agent",
            description="private",
            status=AgentStatus.DRAFT,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)

        resp = client.post(
            "/api/chat/task/create",
            json={
                "title": "agent-ownership-check",
                "description": "desc",
                "agent_id": agent.id,
            },
            headers=user1_headers,
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found or access denied"
    finally:
        db.close()


def test_task_create_rejects_generated_workforce_manager_agent(
    test_db,
    user1_headers,
):
    from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
    from xagent.web.models.database import get_db
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user1 = db.query(User).filter(User.username == "user1").first()
        assert user1 is not None

        agent = Agent(
            user_id=user1.id,
            name="generated-workforce-manager",
            description="private manager",
            status=AgentStatus.PUBLISHED,
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)

        resp = client.post(
            "/api/chat/task/create",
            json={
                "title": "generated-manager-task",
                "description": "desc",
                "agent_id": agent.id,
            },
            headers=user1_headers,
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Agent not found or access denied"

        task = Task(
            user_id=user1.id,
            title="legacy generated manager task",
            description="legacy",
            status=TaskStatus.PENDING,
            agent_id=agent.id,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        assert _load_agent_for_task_runtime(db, task) is None
    finally:
        db.close()


def test_task_create_allows_policy_visible_published_agent(
    test_db, user1_headers, user2_headers
):
    from xagent.web.models.agent import Agent, AgentStatus
    from xagent.web.models.database import get_db
    from xagent.web.models.mcp import MCPServer, UserMCPServer
    from xagent.web.models.task import Task
    from xagent.web.models.user import User

    class VisibleAgentPolicy(WorkforcePolicy):
        def __init__(self, visible_agent_ids: set[int]) -> None:
            self.visible_agent_ids = visible_agent_ids

        def get_visible_agent_ids(self, db, user, purpose: str) -> set[int] | None:
            del db, user
            return self.visible_agent_ids if purpose == "agent_list" else None

    db = next(get_db())
    try:
        user1 = db.query(User).filter(User.username == "user1").first()
        assert user1 is not None
        user2 = db.query(User).filter(User.username == "user2").first()
        assert user2 is not None

        shared_agent = Agent(
            user_id=user2.id,
            name="user2-shared-agent",
            description="shared",
            instructions="Use shared instructions.",
            execution_mode="think",
            status=AgentStatus.PUBLISHED,
            tool_categories=["mcp"],
        )
        db.add(shared_agent)
        owner_server = MCPServer(
            name="owner-runtime-server",
            description="owner connector",
            managed="external",
            transport="streamable_http",
            url="https://example.com/owner/mcp",
            runtime_input_schema={
                "context": {"account_id": {"type": "string", "required": False}}
            },
        )
        actor_server = MCPServer(
            name="actor-runtime-server",
            description="actor connector",
            managed="external",
            transport="streamable_http",
            url="https://example.com/actor/mcp",
            runtime_input_schema={
                "context": {"account_id": {"type": "string", "required": False}}
            },
        )
        db.add_all([owner_server, actor_server])
        db.flush()
        db.add_all(
            [
                UserMCPServer(
                    user_id=user2.id,
                    mcpserver_id=owner_server.id,
                    is_owner=True,
                    is_active=True,
                ),
                UserMCPServer(
                    user_id=user1.id,
                    mcpserver_id=actor_server.id,
                    is_owner=True,
                    is_active=True,
                ),
            ]
        )
        db.commit()
        db.refresh(shared_agent)
        db.refresh(owner_server)
        db.refresh(actor_server)

        set_workforce_policy(VisibleAgentPolicy({int(shared_agent.id)}))

        resp = client.post(
            "/api/chat/task/create",
            json={
                "title": "shared-agent-task",
                "description": "desc",
                "agent_id": shared_agent.id,
            },
            headers=user1_headers,
        )

        assert resp.status_code == 200, resp.text
        task = db.query(Task).filter(Task.id == resp.json()["task_id"]).one()
        assert int(task.agent_id) == int(shared_agent.id)
        assert _load_agent_for_task_runtime(db, task) == shared_agent
        assert task.connector_runtime_selected_refs == [
            {"connector_type": "mcp", "connector_id": int(actor_server.id)}
        ]
        assert {"connector_type": "mcp", "connector_id": int(owner_server.id)} not in (
            task.connector_runtime_selected_refs or []
        )

        from xagent.web.services.task_setup_snapshot import (
            load_task_setup_snapshot_sync,
        )

        snapshot = load_task_setup_snapshot_sync(
            int(task.id),
            task_owner_user_id=int(task.user_id),
        )
        assert snapshot is not None
        assert snapshot.agent is not None
        assert snapshot.agent.id == int(shared_agent.id)
        assert snapshot.excluded_agent_id == int(shared_agent.id)
    finally:
        db.close()


def test_delete_task_removes_trace_blobs_and_task_owned_rows(test_db, user1_headers):
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.database import get_db
    from xagent.web.models.task import (
        DAGExecution,
        DAGExecutionPhase,
        Task,
        TaskStatus,
        TraceCheckpointBlob,
        TraceEvent,
        TraceMessageBlob,
    )
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == "user1").first()
        assert user is not None

        task = Task(
            user_id=user.id,
            title="delete me",
            description="task with task-owned rows",
            status=TaskStatus.COMPLETED,
        )
        db.add(task)
        db.flush()

        db.add_all(
            [
                DAGExecution(
                    task_id=task.id,
                    phase=DAGExecutionPhase.COMPLETED,
                ),
                TraceEvent(
                    task_id=task.id,
                    build_id=None,
                    event_id="vibe-event",
                    event_type="dag_execute_end",
                    timestamp=datetime.now(UTC),
                    data={"ok": True},
                ),
                TraceEvent(
                    task_id=task.id,
                    build_id="builder-session",
                    event_id="build-event",
                    event_type="agent_message",
                    timestamp=datetime.now(UTC),
                    data={"ok": True},
                ),
                TraceMessageBlob(
                    task_id=task.id,
                    execution_id="exec-delete",
                    message_hash="message-hash",
                    message_data={"role": "user", "content": "hello"},
                    message_bytes=42,
                ),
                TraceCheckpointBlob(
                    task_id=task.id,
                    execution_id="exec-delete",
                    blob_kind="messages",
                    blob_hash="checkpoint-hash",
                    blob_data={"messages": []},
                    blob_bytes=17,
                ),
                TaskChatMessage(
                    task_id=task.id,
                    user_id=user.id,
                    role="user",
                    content="hello",
                    message_type="user_message",
                ),
                UploadedFile(
                    user_id=user.id,
                    task_id=task.id,
                    filename="input.txt",
                    storage_path=f"/tmp/task-{task.id}/input.txt",
                    file_size=5,
                ),
            ]
        )
        db.commit()
        task_id = int(task.id)
    finally:
        db.close()

    resp = client.delete(f"/api/chat/task/{task_id}", headers=user1_headers)

    assert resp.status_code == 200, resp.text
    db = next(get_db())
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        assert db.query(TraceMessageBlob).filter_by(task_id=task_id).count() == 0
        assert db.query(TraceCheckpointBlob).filter_by(task_id=task_id).count() == 0
        assert db.query(TraceEvent).filter_by(task_id=task_id).count() == 0
        assert db.query(DAGExecution).filter_by(task_id=task_id).count() == 0
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0
        assert db.query(UploadedFile).filter_by(task_id=task_id).count() == 0
    finally:
        db.close()


def test_delete_task_keeps_cross_user_access_denied(
    test_db, user1_headers, user2_headers
):
    from xagent.web.models.database import get_db
    from xagent.web.models.task import Task
    from xagent.web.models.user import User

    db = next(get_db())
    try:
        user2 = db.query(User).filter(User.username == "user2").first()
        assert user2 is not None
        task = Task(user_id=user2.id, title="private", description="private")
        db.add(task)
        db.commit()
        task_id = int(task.id)
    finally:
        db.close()

    resp = client.delete(f"/api/chat/task/{task_id}", headers=user1_headers)

    assert resp.status_code == 404
    db = next(get_db())
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
    finally:
        db.close()
