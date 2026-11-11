"""Tests for DBOpenShellStore (does not need the openshell SDK)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.sandbox.base import SandboxConfig, SandboxInfo, SandboxTemplate
from xagent.web.models.database import Base
from xagent.web.models.sandbox import SandboxInfo as SandboxInfoModel
from xagent.web.sandbox_store import (
    SANDBOX_TYPE_DOCKER,
    SANDBOX_TYPE_OPENSHELL,
    DBDockerStore,
    DBOpenShellStore,
)


@pytest.fixture
def db_session():  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _store(cls, db_session):  # type: ignore[no-untyped-def]
    store = cls()
    store._get_db_session = lambda: db_session
    return store


def _info(
    name: str = "user::7:client-3", working_dir: str = "/home/sandbox"
) -> SandboxInfo:
    return SandboxInfo(
        name=name,
        state="running",
        template=SandboxTemplate(type="image", image="img:1"),
        config=SandboxConfig(
            working_dir=working_dir,
            cpus=2,
            memory=1024,
            env={"A": "1"},
            volumes=[("/host/ws", "/workspace", "rw"), ("/host/src", "/app/src", "ro")],
            network_isolated=True,
        ),
    )


def test_round_trip_keeps_what_the_gateway_cannot_return(db_session) -> None:  # type: ignore[no-untyped-def]
    store = _store(DBOpenShellStore, db_session)
    store.add_info("user::7:client-3", _info())

    got = store.get_info("user::7:client-3")

    assert got is not None
    assert got.template.image == "img:1"
    assert got.config.working_dir == "/home/sandbox"
    assert got.config.env == {"A": "1"}
    assert got.config.volumes == [
        ("/host/ws", "/workspace", "rw"),
        ("/host/src", "/app/src", "ro"),
    ]
    assert got.config.network_isolated is True
    row = db_session.query(SandboxInfoModel).one()
    assert row.sandbox_type == SANDBOX_TYPE_OPENSHELL


def test_state_update_and_delete(db_session) -> None:  # type: ignore[no-untyped-def]
    store = _store(DBOpenShellStore, db_session)
    store.add_info("n", _info("n"))

    store.update_info_state("n", "stopped")
    assert store.get_info("n").state == "stopped"  # type: ignore[union-attr]

    store.delete_info("n")
    assert store.get_info("n") is None
    # Missing rows are no-ops.
    store.update_info_state("n", "running")
    store.delete_info("n")


def test_add_info_updates_existing_row(db_session) -> None:  # type: ignore[no-untyped-def]
    store = _store(DBOpenShellStore, db_session)
    store.add_info("n", _info("n", working_dir="/a"))
    store.add_info("n", _info("n", working_dir="/b"))

    assert db_session.query(SandboxInfoModel).count() == 1
    assert store.get_info("n").config.working_dir == "/b"  # type: ignore[union-attr]


def test_rows_are_isolated_from_docker_backend(db_session) -> None:  # type: ignore[no-untyped-def]
    """Same sandbox name under two backends must not collide or leak."""
    openshell = _store(DBOpenShellStore, db_session)
    docker = _store(DBDockerStore, db_session)
    openshell.add_info("n", _info("n", working_dir="/openshell"))
    docker.add_info("n", _info("n", working_dir="/docker"))

    assert openshell.get_info("n").config.working_dir == "/openshell"  # type: ignore[union-attr]
    assert docker.get_info("n").config.working_dir == "/docker"  # type: ignore[union-attr]

    openshell.delete_info("n")
    assert openshell.get_info("n") is None
    assert docker.get_info("n") is not None
    assert {r.sandbox_type for r in db_session.query(SandboxInfoModel)} == {
        SANDBOX_TYPE_DOCKER
    }
