"""PostgreSQL concurrency coverage for runtime-key transition fencing."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from xagent.web.models.agent import Agent
from xagent.web.models.agent_api_key import AgentApiKey
from xagent.web.models.database import (
    Base,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.user import User
from xagent.web.services.agent_management import AgentManagementRuntime
from xagent.web.services.api_keys import (
    AgentApiKeyService,
    RuntimeKeyReceipt,
    acquire_runtime_key_transition_fence,
)


@pytest.fixture()
def postgres_runtime_keys():
    """Isolated runtime-key rows in a real PostgreSQL test database."""

    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = get_session_local()
    try:
        with SessionLocal() as db:
            user = User(
                username="runtime-key-transition",
                password_hash="hash",
                is_admin=True,
            )
            db.add(user)
            db.flush()
            agent = Agent(
                user_id=int(user.id),
                name="runtime-key-transition",
                instructions="test",
            )
            db.add(agent)
            db.flush()
            agent_id = int(agent.id)
            db.add(
                AgentApiKey(
                    agent_id=agent_id,
                    key_prefix="ORIGIN",
                    key_hash="origin-hash",
                )
            )
            db.commit()
        yield SessionLocal, agent_id
    finally:
        Base.metadata.drop_all(bind=engine)


def test_compensation_serializes_with_a_later_postgres_rotation(
    postgres_runtime_keys,
) -> None:
    """A later rotation snapshot must commit before stale compensation runs."""

    SessionLocal, agent_id = postgres_runtime_keys
    with SessionLocal() as db:
        service = AgentApiKeyService(db)
        service.rotate_key_for_runtime_delivery(
            agent_id,
            candidate=("xag_FIRST1_" + "a" * 32, "FIRST1", "first-hash"),
        )
        receipt = service.runtime_key_receipt
        assert isinstance(receipt, RuntimeKeyReceipt)

    snapshot_taken = threading.Event()
    release_rotation = threading.Event()
    compensation_started = threading.Event()
    compensation_finished = threading.Event()

    def run_later_rotation() -> None:
        with SessionLocal() as db:
            assert acquire_runtime_key_transition_fence(db, agent_id)
            replaced_key_ids = tuple(
                row_id
                for (row_id,) in (
                    db.query(AgentApiKey.id)
                    .filter(
                        AgentApiKey.agent_id == agent_id,
                        AgentApiKey.revoked_at.is_(None),
                    )
                    .all()
                )
            )
            snapshot_taken.set()
            assert release_rotation.wait(timeout=10)
            now = datetime.now(timezone.utc)
            revoked = (
                db.query(AgentApiKey)
                .filter(
                    AgentApiKey.id.in_(replaced_key_ids),
                    AgentApiKey.revoked_at.is_(None),
                )
                .update(
                    {
                        AgentApiKey.revoked_at: now,
                        AgentApiKey.updated_at: now,
                    },
                    synchronize_session=False,
                )
            )
            assert revoked == 1
            db.add(
                AgentApiKey(
                    agent_id=agent_id,
                    key_prefix="SECOND",
                    key_hash="second-hash",
                )
            )
            db.commit()

    def run_compensation():
        compensation_started.set()
        try:
            return AgentManagementRuntime._compensate_runtime_key_sync(receipt)
        finally:
            compensation_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        rotation = executor.submit(run_later_rotation)
        assert snapshot_taken.wait(timeout=10)
        compensation = executor.submit(run_compensation)
        assert compensation_started.wait(timeout=10)
        try:
            assert not compensation_finished.wait(timeout=0.2)
        finally:
            release_rotation.set()
        rotation.result(timeout=10)
        compensation_result = compensation.result(timeout=10)

    assert compensation_result.new_key_revoked == 0
    assert compensation_result.prior_keys_restored == 0
    with SessionLocal() as db:
        rows = (
            db.query(AgentApiKey)
            .filter(AgentApiKey.agent_id == agent_id)
            .order_by(AgentApiKey.id)
            .all()
        )
        assert [row.key_prefix for row in rows if row.revoked_at is None] == ["SECOND"]
