import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.api.agents import router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User
from xagent.web.services.agent_host_settings import set_agent_host_settings_writer

host_rows = Table(
    "test_agent_host_settings",
    MetaData(),
    Column("agent_id", Integer, primary_key=True),
    Column("policy", String, nullable=False),
)


@pytest.fixture
def api():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    host_rows.create(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions.begin() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)

    def session():
        with sessions() as db:
            yield db

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = session
    set_agent_host_settings_writer(None)
    try:
        with TestClient(app) as client:
            yield client, sessions
    finally:
        set_agent_host_settings_writer(None)
        engine.dispose()


def _stage_policy(db: Session, agent: Agent, user: User, settings: dict):
    assert agent.user_id == user.id
    db.execute(host_rows.insert().values(agent_id=agent.id, policy=settings["policy"]))


def _reject_policy(db: Session, agent: Agent, user: User, settings: dict):
    _stage_policy(db, agent, user, settings)
    raise HTTPException(422, "Invalid policy")


def test_create_rejects_nonempty_settings_without_a_writer(api):
    client, sessions = api
    assert (
        client.post(
            "/api/agents",
            json={"name": "Rejected", "host_settings": {"policy": "strict"}},
        ).status_code
        == 422
    )
    with sessions() as db:
        assert db.scalars(select(Agent)).all() == []
    assert (
        client.post(
            "/api/agents", json={"name": "Accepted", "host_settings": {}}
        ).status_code
        == 200
    )


def test_writer_failure_rolls_back_create(api):
    client, sessions = api
    set_agent_host_settings_writer(_reject_policy)
    response = client.post(
        "/api/agents", json={"name": "Rejected", "host_settings": {"policy": "strict"}}
    )
    assert response.status_code == 422, response.text
    with sessions() as db:
        assert db.scalars(select(Agent)).all() == []
        assert db.execute(select(host_rows)).all() == []


def test_writer_failure_rolls_back_update(api):
    client, sessions = api
    agent_id = client.post("/api/agents", json={"name": "Original"}).json()["id"]
    set_agent_host_settings_writer(_reject_policy)
    response = client.put(
        f"/api/agents/{agent_id}",
        json={"name": "Renamed", "host_settings": {"policy": "strict"}},
    )
    assert response.status_code == 422, response.text
    with sessions() as db:
        assert db.get(Agent, agent_id).name == "Original"
        assert db.execute(select(host_rows)).all() == []


def test_host_settings_only_update_commits_staged_data(api):
    client, sessions = api
    agent_id = client.post("/api/agents", json={"name": "Original"}).json()["id"]
    set_agent_host_settings_writer(_stage_policy)
    response = client.put(
        f"/api/agents/{agent_id}", json={"host_settings": {"policy": "strict"}}
    )
    assert response.status_code == 200, response.text
    with sessions() as db:
        assert db.execute(select(host_rows)).all() == [(agent_id, "strict")]
