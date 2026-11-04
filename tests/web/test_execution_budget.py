from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.budget import ExecutionBudgetPolicy
from xagent.web.api.auth import UpdatePreferencesRequest
from xagent.web.api.execution_budget import router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User
from xagent.web.services import execution_budget as budget_module
from xagent.web.services.execution_budget import (
    BudgetPolicyRequest,
    ExecutionBudgetDefaults,
    resolve_default_policy,
    resolve_execution_budget_policy,
    set_execution_budget_policy_resolver,
)


@pytest.fixture
def budget_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'budget.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(
            User(
                id=1,
                username="budget-owner",
                password_hash="unused",
                preferences={
                    "execution_budget_tokens": 200,
                    "execution_budget_soft_percent": 65,
                },
            )
        )
        db.commit()
    monkeypatch.setattr(budget_module, "get_session_local", lambda: factory)
    set_execution_budget_policy_resolver(None)
    yield factory
    set_execution_budget_policy_resolver(None)
    engine.dispose()


@pytest.fixture
def budget_client(budget_db):
    app = FastAPI()
    app.include_router(router)
    user = SimpleNamespace(
        id=1,
        is_admin=False,
        preferences={
            "execution_budget_tokens": 200,
            "execution_budget_soft_percent": 65,
        },
    )

    def get_test_db():
        with budget_db() as db:
            yield db

    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as client:
        yield client, user


def test_defaults_inherit_and_user_override_cannot_exceed_maximum():
    defaults = ExecutionBudgetDefaults(
        default_tokens=100, max_tokens=300, soft_limit_percent=70
    )
    assert resolve_default_policy(defaults, {}).max_tokens == 100
    override = resolve_default_policy(
        defaults, {"execution_budget_tokens": 200, "execution_budget_soft_percent": 60}
    )
    assert override.max_tokens == 200
    assert override.source == "personal"
    assert override.soft_limit_percent == 60
    clipped = resolve_default_policy(defaults, {"execution_budget_tokens": 1000})
    assert clipped.max_tokens == 300
    assert clipped.source == "system_maximum"
    assert (
        resolve_default_policy(ExecutionBudgetDefaults(max_tokens=150), {}).max_tokens
        == 150
    )


@pytest.mark.asyncio
async def test_external_resolver_receives_trusted_identity_without_replacing_quota_hook(
    budget_db,
):
    from xagent.web.services import quota_hooks

    calls = []

    async def resolver(request, policy):
        calls.append((request, policy))
        return ExecutionBudgetPolicy(
            max_tokens=50, soft_limit_percent=45, source="workspace"
        )

    set_execution_budget_policy_resolver(resolver)
    original = quota_hooks._run_progress_gate_hook
    quota_hooks.set_run_progress_gate_hook(lambda *args: "existing quota")
    try:
        request = BudgetPolicyRequest(user_id=1, task_id="task-123")
        policy = await resolve_execution_budget_policy(request)
        assert policy.max_tokens == 50
        assert policy.source == "workspace"
        assert calls[0][0] == request
        assert calls[0][1].max_tokens == 200
        assert quota_hooks.check_run_progress_gate(None, 1, [], 0) == "existing quota"
    finally:
        quota_hooks.set_run_progress_gate_hook(original)


@pytest.mark.asyncio
async def test_resolver_failure_is_not_unlimited(budget_db):
    def resolver(*args):
        raise RuntimeError("policy lookup unavailable")

    set_execution_budget_policy_resolver(resolver)
    with pytest.raises(RuntimeError, match="policy lookup unavailable"):
        await resolve_execution_budget_policy(BudgetPolicyRequest(user_id=1))


def test_admin_defaults_and_current_effective_policy(budget_client):
    client, user = budget_client
    settings = {"default_tokens": 100, "max_tokens": 150, "soft_limit_percent": 80}
    assert (
        client.put("/api/execution-budget/defaults", json=settings).status_code == 403
    )
    assert client.get("/api/execution-budget/defaults").status_code == 403
    user.is_admin = True
    assert (
        client.put("/api/execution-budget/defaults", json=settings).status_code == 200
    )
    assert client.get("/api/execution-budget/defaults").json() == settings
    user.is_admin = False
    result = client.get("/api/execution-budget/me")
    assert result.status_code == 200
    assert result.json()["effective"]["max_tokens"] == 150
    assert result.json()["effective"]["soft_limit_percent"] == 65
    assert result.json()["effective"]["soft_limit_source"] == "personal"
    assert result.json()["preferences"]["execution_budget_tokens"] == 200


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_personal_limit_validation(value):
    with pytest.raises(ValidationError):
        UpdatePreferencesRequest(execution_budget_tokens=value)


def test_null_personal_settings_mean_inherit():
    assert UpdatePreferencesRequest(
        execution_budget_tokens=None, execution_budget_soft_percent=None
    ).model_dump(exclude_unset=True) == {
        "execution_budget_tokens": None,
        "execution_budget_soft_percent": None,
    }


@pytest.mark.asyncio
async def test_provider_cannot_relax_system_maximum(budget_db, monkeypatch):
    monkeypatch.setenv("XAGENT_EXECUTION_BUDGET_MAX_TOKENS", "80")
    set_execution_budget_policy_resolver(
        lambda *args: ExecutionBudgetPolicy(max_tokens=None)
    )
    result = await resolve_execution_budget_policy(BudgetPolicyRequest(user_id=1))
    assert result.max_tokens == 80


def test_admin_default_cannot_exceed_maximum(budget_client):
    client, user = budget_client
    user.is_admin = True
    result = client.put(
        "/api/execution-budget/defaults",
        json={"default_tokens": 200, "max_tokens": 100, "soft_limit_percent": 80},
    )
    assert result.status_code == 422
    assert "default_tokens must not exceed max_tokens" in result.text


def test_saved_admin_settings_replace_all_environment_defaults(
    budget_client, monkeypatch
):
    client, user = budget_client
    user.is_admin = True
    saved = {"default_tokens": 120, "max_tokens": None, "soft_limit_percent": 70}
    assert client.put("/api/execution-budget/defaults", json=saved).status_code == 200
    monkeypatch.setenv("XAGENT_EXECUTION_BUDGET_DEFAULT_TOKENS", "50")
    monkeypatch.setenv("XAGENT_EXECUTION_BUDGET_MAX_TOKENS", "80")
    monkeypatch.setenv("XAGENT_EXECUTION_BUDGET_SOFT_PERCENT", "40")
    assert client.get("/api/execution-budget/defaults").json() == saved
    effective = client.get("/api/execution-budget/me").json()["effective"]
    assert effective["max_tokens"] == 200
