from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.api.auth import auth_router
from xagent.web.first_admin_setup import (
    FirstAdminIdentity,
    register_first_admin_setup_hook,
)
from xagent.web.models.database import Base, get_db
from xagent.web.models.system_setting import SystemSetting
from xagent.web.models.user import User

ADMIN = {
    "username": "admin",
    "email": "admin@example.com",
    "password": "admin123",
}


@pytest.fixture
def app_factory() -> Iterator:
    engines = []

    def make_app() -> tuple[FastAPI, sessionmaker[Session]]:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        engines.append(engine)
        Base.metadata.create_all(engine)
        session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        def override_get_db() -> Iterator[Session]:
            with session_local() as db:
                yield db

        app = FastAPI()
        app.include_router(auth_router)
        app.dependency_overrides[get_db] = override_get_db
        return app, session_local

    yield make_app
    for engine in engines:
        engine.dispose()


def _post_setup(app: FastAPI, *, raise_server_exceptions: bool = True):
    with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
        return client.post("/api/auth/setup-admin", json=ADMIN)


def test_hook_stages_with_core_setup_in_one_commit(app_factory):
    app, session_local = app_factory()
    commits = 0
    observed: dict[str, object] = {}

    def hook(db: Session, admin: FirstAdminIdentity) -> None:
        observed["session"] = db
        observed["admin"] = admin
        assert db.get(User, admin.user_id).username == ADMIN["username"]
        assert any(
            isinstance(row, SystemSetting)
            and row.key == "setup_completed"
            and row.value == "true"
            for row in db.new
        )
        db.add(SystemSetting(key="host_admin_id", value=str(admin.user_id)))

    def count_commit(_session: Session) -> None:
        nonlocal commits
        commits += 1

    register_first_admin_setup_hook(app, hook)
    event.listen(Session, "after_commit", count_commit)
    try:
        response = _post_setup(app)
    finally:
        event.remove(Session, "after_commit", count_commit)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert commits == 1
    assert isinstance(observed["admin"], FirstAdminIdentity)
    with session_local() as db:
        admin = db.query(User).one()
        assert (
            db.query(SystemSetting).filter_by(key="setup_completed").one().value
            == "true"
        )
        assert db.query(SystemSetting).filter_by(
            key="host_admin_id"
        ).one().value == str(admin.id)


def test_hook_failure_rolls_back_core_and_hook_rows(app_factory):
    app, session_local = app_factory()

    def failing_hook(db: Session, admin: FirstAdminIdentity) -> None:
        db.add(SystemSetting(key="host_admin_id", value=str(admin.user_id)))
        db.flush()
        raise RuntimeError("host setup failed")

    register_first_admin_setup_hook(app, failing_hook)
    response = _post_setup(app, raise_server_exceptions=False)

    assert response.status_code == 500
    with session_local() as db:
        assert db.query(User).count() == 0
        assert db.query(SystemSetting).count() == 0


def test_hook_registration_is_isolated_by_app(app_factory):
    hooked_app, hooked_sessions = app_factory()
    plain_app, plain_sessions = app_factory()

    register_first_admin_setup_hook(
        hooked_app,
        lambda db, admin: db.add(
            SystemSetting(key="host_admin_id", value=str(admin.user_id))
        ),
    )

    assert _post_setup(plain_app).json()["success"] is True
    assert _post_setup(hooked_app).json()["success"] is True
    with plain_sessions() as db:
        assert (
            db.query(SystemSetting).filter_by(key="host_admin_id").one_or_none() is None
        )
    with hooked_sessions() as db:
        assert (
            db.query(SystemSetting).filter_by(key="host_admin_id").one_or_none()
            is not None
        )


def test_setup_without_hook_remains_compatible(app_factory):
    app, session_local = app_factory()

    response = _post_setup(app)

    assert response.status_code == 200
    assert response.json()["success"] is True
    with session_local() as db:
        assert db.query(User).filter_by(is_admin=True).count() == 1
        assert (
            db.query(SystemSetting).filter_by(key="setup_completed").one().value
            == "true"
        )
