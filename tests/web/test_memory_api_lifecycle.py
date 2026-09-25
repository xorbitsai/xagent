"""Memory API semantics across every persistent-memory lifecycle state.

The point of these tests is the pairing: when memory is fenced off, memory
routes fail closed with one stable, public-safe 503, while unrelated routes and
task orchestration keep working.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.web.api.auth import auth_router, hash_password
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.memory_lifecycle import (
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager as agent_runtime_service

_TEMP_DIR = tempfile.mkdtemp()
_ENGINE = create_engine(
    f"sqlite:///{os.path.join(_TEMP_DIR, 'memory-lifecycle-api.db')}",
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)

UNAVAILABLE_STATES = [
    MemoryLifecycleState.BLOCKED_REPAIR,
    MemoryLifecycleState.RESTART_REQUIRED,
    MemoryLifecycleState.CREDENTIAL_UNAVAILABLE,
    MemoryLifecycleState.RETRYABLE_UNAVAILABLE,
]


def _override_get_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_user():
    Base.metadata.create_all(bind=_ENGINE)
    session = _SessionLocal()
    try:
        user = User(
            username="memory-lifecycle-admin",
            password_hash=hash_password("admin"),
            is_admin=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        yield user
    finally:
        session.close()
        Base.metadata.drop_all(bind=_ENGINE)


@pytest.fixture
def auth_headers(test_user):
    token = jwt.encode(
        {
            "sub": test_user.username,
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "user_id": test_user.id,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}


def _client(provider) -> TestClient:
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(MemoryManagementRouter(provider).get_router())

    @app.get("/api/unrelated/health")
    def unrelated() -> dict:
        """Stands in for every route that has nothing to do with memory."""
        return {"ok": True}

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def _unavailable(state: MemoryLifecycleState):
    status = MemoryLifecycleStatus(state)

    def provider():
        raise MemoryUnavailableError(status)

    return provider, status


@pytest.mark.parametrize("state", UNAVAILABLE_STATES)
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/memory/list", None),
        ("get", "/api/memory/stats", None),
        ("get", "/api/memory/some-id", None),
        ("post", "/api/memory", {"content": "note"}),
        ("put", "/api/memory/some-id", {"content": "note"}),
        ("delete", "/api/memory/some-id", None),
    ],
)
def test_every_memory_route_fails_closed_with_one_stable_detail(
    auth_headers, state, method, path, body
):
    provider, status = _unavailable(state)
    client = _client(provider)

    response = getattr(client, method)(
        path, headers=auth_headers, **({"json": body} if body is not None else {})
    )

    assert response.status_code == 503
    assert response.json()["detail"] == status.detail


@pytest.mark.parametrize("state", UNAVAILABLE_STATES)
def test_unrelated_routes_keep_working_while_memory_is_fenced_off(auth_headers, state):
    provider, _status = _unavailable(state)
    client = _client(provider)

    assert client.get("/api/unrelated/health").json() == {"ok": True}
    # The 503 (not a 401) also shows authentication still resolves normally
    # while memory itself is closed.
    assert client.get("/api/memory/list", headers=auth_headers).status_code == 503


class _RevokedAfterAcquire:
    """A published proxy revoked between acquisition and the operation.

    This is what the runtime actually hands out: the store the router acquires
    is a revocation proxy that re-checks its publication generation on *every*
    call, so a concurrent revalidation raises out of the delegated operation
    rather than out of the acquisition the router already mapped.
    """

    def __init__(self, status: MemoryLifecycleStatus) -> None:
        self._status = status

    def _revoked(self, *_args, **_kwargs):
        raise MemoryUnavailableError(self._status)

    add = _revoked
    get = _revoked
    update = _revoked
    delete = _revoked
    search = _revoked
    list_all = _revoked
    get_stats = _revoked


@pytest.mark.parametrize("state", UNAVAILABLE_STATES)
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/memory/list", None),
        ("get", "/api/memory/list?search=note", None),
        ("get", "/api/memory/stats", None),
        ("get", "/api/memory/some-id", None),
        ("post", "/api/memory", {"content": "note"}),
        ("put", "/api/memory/some-id", {"content": "note"}),
        ("delete", "/api/memory/some-id", None),
    ],
)
def test_a_revocation_after_acquisition_still_answers_the_stable_503(
    auth_headers, state, method, path, body
):
    """Revocation mid-operation is the same public condition as at acquisition.

    Each route's broad ``except Exception`` used to turn the delegated
    ``MemoryUnavailableError`` into a 500 naming the internal failure, so the
    stable 503 depended on losing a race with revalidation.
    """
    status = MemoryLifecycleStatus(state)
    client = _client(lambda: _RevokedAfterAcquire(status))

    response = getattr(client, method)(
        path, headers=auth_headers, **({"json": body} if body is not None else {})
    )

    assert response.status_code == 503
    assert response.json()["detail"] == status.detail


def test_store_info_keeps_answering_two_hundred_through_a_revocation(
    monkeypatch, auth_headers
):
    """The one route that must not turn a revocation into a 503."""
    from xagent.web.api import memory as memory_api

    info = {
        "store_type": None,
        "is_lancedb": False,
        "state": MemoryLifecycleState.RESTART_REQUIRED.value,
        "detail": MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED).detail,
        "mode": None,
        "supports_vector_search": False,
        "similarity_threshold": 1.5,
    }

    class _Manager:
        def get_store_info(self) -> dict:
            return info

    monkeypatch.setattr(memory_api, "get_memory_store_manager", lambda: _Manager())
    status = MemoryLifecycleStatus(MemoryLifecycleState.RESTART_REQUIRED)
    client = _client(lambda: _RevokedAfterAcquire(status))

    response = client.get("/api/memory/store-info", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == info


def test_store_info_answers_while_memory_is_blocked(monkeypatch, auth_headers):
    """The status route must survive the state it exists to report."""
    from xagent.web.api import memory as memory_api

    blocked = {
        "store_type": None,
        "is_lancedb": False,
        "state": MemoryLifecycleState.BLOCKED_REPAIR.value,
        "detail": MemoryLifecycleStatus(MemoryLifecycleState.BLOCKED_REPAIR).detail,
        "mode": None,
        "supports_vector_search": False,
        "similarity_threshold": 1.5,
    }

    class _Manager:
        def get_store_info(self) -> dict:
            return blocked

    monkeypatch.setattr(memory_api, "get_memory_store_manager", lambda: _Manager())
    provider, _status = _unavailable(MemoryLifecycleState.BLOCKED_REPAIR)

    response = _client(provider).get("/api/memory/store-info", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == blocked


def test_store_info_is_not_shadowed_by_the_memory_id_route(monkeypatch, auth_headers):
    """``/store-info`` must not be answered as a lookup for a note by that id."""
    seen: list[str] = []

    class _Store:
        def get(self, memory_id):
            seen.append(memory_id)
            raise AssertionError("store-info must not reach the note lookup")

    class _Manager:
        def get_store_info(self) -> dict:
            return {"state": "ready"}

    from xagent.web.api import memory as memory_api

    monkeypatch.setattr(memory_api, "get_memory_store_manager", lambda: _Manager())
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(MemoryManagementRouter(lambda: _Store()).get_router())
    app.dependency_overrides[get_db] = _override_get_db

    response = TestClient(app).get("/api/memory/store-info", headers=auth_headers)

    assert seen == []
    assert response.status_code == 200
    assert response.json()["state"] == "ready"


@pytest.mark.parametrize("state", UNAVAILABLE_STATES)
def test_task_orchestration_starts_with_memory_disabled(monkeypatch, state):
    """A fenced-off memory disables memory, it does not stop the task."""
    status = MemoryLifecycleStatus(state)

    def provider():
        raise MemoryUnavailableError(status)

    monkeypatch.setattr(agent_runtime_service, "get_memory_store", provider)

    policy = agent_runtime_service.resolve_agent_service_memory_policy(agent_config={})

    assert policy.memory_enabled is False
    assert policy.memory_available is False
    assert policy.memory_availability_reason == state.value
    # Inert placeholder: nothing reaches the storage admission refused.
    assert isinstance(policy.memory, InMemoryMemoryStore)


def test_task_orchestration_uses_the_published_store_when_ready(monkeypatch):
    store = InMemoryMemoryStore()
    monkeypatch.setattr(agent_runtime_service, "get_memory_store", lambda: store)

    policy = agent_runtime_service.resolve_agent_service_memory_policy(agent_config={})

    assert policy.memory is store
    assert policy.memory_enabled is True
    assert policy.memory_available is True
    assert policy.memory_availability_reason is None
