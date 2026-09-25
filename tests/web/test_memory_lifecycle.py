"""Layer D: startup admission, publication and drift for persistent memory.

These tests pin the four owner contracts:

* only the explicit global authority is consumed,
* invalid legacy data fails closed as BLOCKED_REPAIR while unrelated functions
  keep working,
* nothing reloads online -- meaningful drift demands a restart,
* and no store is ever published before admission certifies one.
"""

from __future__ import annotations

import ast
import json
import multiprocessing
import pathlib
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest
from fastapi import FastAPI
from filelock import FileLock
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import (
    EXHAUSTION_POOL_TIMEOUT,
    assert_pool_checkout_off_loop,
)
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.memory.lancedb_maintenance import lancedb_lock_path
from xagent.core.memory.storage_admission import MemoryStorageMode
from xagent.core.memory.vector_compatibility import (
    canonical_embedding_identity,
    embedding_identity_fingerprint,
)
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.providers.vector_store.lancedb import clear_connection_cache
from xagent.web import dynamic_memory_store as manager_module
from xagent.web import memory_lifecycle
from xagent.web import memory_utils as memory_utils_module
from xagent.web.dynamic_memory_store import (
    AuthorityUnreadable,
    DynamicMemoryStoreManager,
)
from xagent.web.memory_lifecycle import (
    MEMORY_TABLE_NAME,
    MemoryLifecycleState,
    MemoryUnavailableError,
    admit_authority_storage,
    authority_embedding_config,
)
from xagent.web.models.global_memory_embedding_authority import (
    GlobalMemoryEmbeddingAuthority,
)
from xagent.web.revocable_memory_store import (
    RevocableMemoryStore,
    unwrap_memory_store,
)
from xagent.web.services import agent_service_manager
from xagent.web.services.global_memory_embedding_authority import (
    CREDENTIAL_CONFIGURED,
    AuthorityConfiguration,
    AuthorityCredentialUnavailable,
    CredentialSource,
    GlobalMemoryEmbeddingAuthorityService,
    GlobalMemoryEmbeddingAuthoritySnapshot,
)
from xagent.web.user_isolated_memory import UserContext, UserIsolatedMemoryStore

DIMENSION = 4


class ConstantEmbedding(BaseEmbedding):
    """Deterministic in-process embedding.

    Layer D is exercised with this rather than a live provider: no real
    endpoint is reachable from the test environment, and fabricating one would
    claim coverage the suite does not have.
    """

    def __init__(self, seed: float = 0.5) -> None:
        self._seed = seed

    def encode(self, text, dimension=None, instruct=None):
        vector = [self._seed] * DIMENSION
        return vector if isinstance(text, str) else [vector for _ in text]

    def get_dimension(self):
        return DIMENSION

    @property
    def abilities(self):
        return ["embed"]


def _snapshot(
    *,
    model_name: str = "text-embedding-3-small",
    dimension: int = DIMENSION,
    api_key: str = "secret-key",
    credential_identity: str = "identity-1",
    max_retries: int = 3,
) -> GlobalMemoryEmbeddingAuthoritySnapshot:
    now = datetime.now(timezone.utc)
    return GlobalMemoryEmbeddingAuthoritySnapshot(
        provider="openai",
        model_name=model_name,
        endpoint="https://api.openai.com/v1/embeddings",
        dimension=dimension,
        instruct=None,
        max_retries=max_retries,
        credential_source=CredentialSource.ORGANIZATION_OWNED,
        global_sharing_consent=True,
        consented_by_actor_subject="admin-subject",
        consented_at=now,
        created_at=now,
        updated_at=now,
        credential_status=CREDENTIAL_CONFIGURED,
        api_key=SecretStr(api_key),
        credential_identity=credential_identity,
    )


def _admit(tmp_path, snapshot=None, **kwargs):
    return admit_authority_storage(
        snapshot if snapshot is not None else _snapshot(),
        db_dir=str(tmp_path),
        embedding_factory=lambda _config: ConstantEmbedding(),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _isolated_lancedb():
    clear_connection_cache()
    yield
    clear_connection_cache()


def _install_authority(
    monkeypatch,
    snapshot: Optional[GlobalMemoryEmbeddingAuthoritySnapshot],
    *,
    error: Optional[BaseException] = None,
) -> dict[str, Any]:
    """Point the manager's authority read at a fixed answer."""
    state: dict[str, Any] = {"snapshot": snapshot, "error": error, "reads": 0}

    def read() -> Optional[GlobalMemoryEmbeddingAuthoritySnapshot]:
        state["reads"] += 1
        if state["error"] is not None:
            raise state["error"]
        return state["snapshot"]

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read)
    return state


def _manager(monkeypatch, tmp_path, state_holder=None):
    """A manager whose admission lands in ``tmp_path`` with a fake embedding."""
    real = memory_lifecycle.admit_authority_storage

    def admit(snapshot, **kwargs):
        kwargs["db_dir"] = str(tmp_path)
        kwargs["embedding_factory"] = lambda _config: ConstantEmbedding()
        return real(snapshot, **kwargs)

    monkeypatch.setattr(manager_module, "admit_authority_storage", admit)
    return DynamicMemoryStoreManager()


def _seed_invalid_table(tmp_path) -> None:
    """A legacy table with a NULL id, which admission must refuse to touch."""
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        MEMORY_TABLE_NAME,
        pa.table(
            {
                "id": ["note-0", None],
                "text": ["first", "second"],
                "metadata": [json.dumps({"user_id": 1})] * 2,
            }
        ),
    )
    _safe_close_table(table)


# --------------------------------------------------------------------------
# The authority is the only source of runtime identity.
# --------------------------------------------------------------------------


def test_runtime_identity_comes_from_the_authority_alone():
    snapshot = _snapshot()
    config = authority_embedding_config(snapshot)

    assert config.model_provider == "openai"
    assert config.model_name == "text-embedding-3-small"
    assert config.base_url == snapshot.endpoint
    assert config.dimension == DIMENSION
    assert config.max_retries == snapshot.max_retries
    assert config.api_key == "secret-key"


def test_manager_never_reads_the_model_hub_or_personal_defaults():
    """The forbidden personal-default resolution has no surface left."""
    assert not hasattr(DynamicMemoryStoreManager, "_get_embedding_model_from_db")
    assert not hasattr(DynamicMemoryStoreManager, "_create_lancedb_store")

    # Nothing on the runtime path imports the personal-default or model-hub
    # machinery, so there is no code by which it could be consulted.
    for module in (memory_lifecycle, manager_module, memory_utils_module):
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "UserDefaultModel" not in imported
        assert "SQLAlchemyModelHub" not in imported
        assert "EmbeddingModelConfig" not in imported or module is memory_lifecycle


# --------------------------------------------------------------------------
# Publication happens only after admission succeeds.
# --------------------------------------------------------------------------


def test_admission_publishes_a_vector_capable_store(tmp_path):
    result = _admit(tmp_path)

    assert result.status.state is MemoryLifecycleState.READY
    assert result.status.mode is MemoryStorageMode.VECTOR
    assert result.status.vector_search is True
    assert isinstance(result.store, UserIsolatedMemoryStore)
    assert result.vector_space_fingerprint == _snapshot().vector_space_fingerprint()


def test_invalid_legacy_data_blocks_repair_and_publishes_nothing(tmp_path):
    _seed_invalid_table(tmp_path)
    before = lancedb.connect(tmp_path).open_table(MEMORY_TABLE_NAME).to_arrow()

    result = _admit(tmp_path)

    assert result.status.state is MemoryLifecycleState.BLOCKED_REPAIR
    assert result.store is None
    assert result.vector_space_fingerprint is None
    clear_connection_cache()
    after = lancedb.connect(tmp_path).open_table(MEMORY_TABLE_NAME).to_arrow()
    assert after == before


def test_contended_admission_is_retryable_and_publishes_nothing(tmp_path):
    connection = lancedb.connect(tmp_path)
    started = time.monotonic()
    with FileLock(lancedb_lock_path(connection, MEMORY_TABLE_NAME, "admission")):
        result = _admit(tmp_path, lock_timeout=0.05)
    assert time.monotonic() - started < 10

    assert result.status.state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    assert result.store is None


def test_quiescence_is_required_before_anything_is_published(tmp_path):
    result = _admit(tmp_path, writers_quiesced=False)

    assert result.status.state is MemoryLifecycleState.RESTART_REQUIRED
    assert result.store is None


def test_unbounded_lock_timeout_is_a_caller_bug(tmp_path):
    with pytest.raises(ValueError):
        _admit(tmp_path, lock_timeout=-1)


def test_store_construction_failure_publishes_nothing(tmp_path):
    def explode(_config):
        raise OSError("adapter unavailable")

    result = admit_authority_storage(
        _snapshot(), db_dir=str(tmp_path), embedding_factory=explode
    )

    assert result.status.state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    assert result.store is None


def test_manager_publishes_nothing_until_admission_succeeds(monkeypatch, tmp_path):
    _seed_invalid_table(tmp_path)
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)

    assert manager._publication is None
    status = manager.admit()

    assert status.state is MemoryLifecycleState.BLOCKED_REPAIR
    assert manager._publication is None
    with pytest.raises(MemoryUnavailableError) as raised:
        manager.get_memory_store()
    assert raised.value.status.state is MemoryLifecycleState.BLOCKED_REPAIR


def test_importing_the_store_module_publishes_nothing(monkeypatch):
    """The module-level singleton is gone; the name resolves on access."""
    import importlib

    module = importlib.import_module("xagent.web.memory_store")
    importlib.reload(module)

    calls: list[int] = []

    def provider():
        calls.append(1)
        return "store"

    monkeypatch.setattr(module, "get_memory_store", provider)
    assert calls == []
    assert module.global_memory_store == "store"
    assert module.base_memory_store == "store"
    assert calls == [1, 1]
    with pytest.raises(AttributeError):
        module.something_else


# --------------------------------------------------------------------------
# One canonical embedding identity.
# --------------------------------------------------------------------------

#: Minimal dashscope authority request. Dashscope is the one provider that
#: keeps ``instruct`` at all, so it is the only place a spelling of that field
#: can reach persistence and be mistaken for a different vector space.
_DASHSCOPE_AUTHORITY = {
    "provider": "dashscope",
    "model_name": "text-embedding-v4",
    "dimension": DIMENSION,
    "max_retries": 3,
    "credential_source": "organization_owned",
    "global_sharing_consent": True,
    "api_key": "organization-owned-secret",
}


@pytest.fixture
def authority_database(tmp_path):
    """A real authority table, so canonicalization runs on the way in and out."""
    engine = create_engine(f"sqlite:///{tmp_path / 'authority.db'}")
    GlobalMemoryEmbeddingAuthority.__table__.create(engine)
    try:
        yield sessionmaker(bind=engine)
    finally:
        engine.dispose()


def _store_authority(sessions, **overrides) -> None:
    with sessions() as db:
        GlobalMemoryEmbeddingAuthorityService(db).set(
            AuthorityConfiguration(**{**_DASHSCOPE_AUTHORITY, **overrides}),
            actor_subject="admin-subject",
        )


def _persisted_instruct(sessions):
    with sessions() as db:
        return db.get(GlobalMemoryEmbeddingAuthority, "global").instruct


def _write_raw_instruct(sessions, value) -> None:
    """Plant a row an earlier build could have written, bypassing the service."""
    with sessions() as db:
        db.get(GlobalMemoryEmbeddingAuthority, "global").instruct = value
        db.commit()


def _loaded_snapshot(sessions):
    with sessions() as db:
        return GlobalMemoryEmbeddingAuthorityService(db).load_snapshot()


def test_one_vector_space_fingerprints_once_however_it_is_spelled(tmp_path):
    """Admission keys on the canonical identity, not on how it was written.

    Fingerprinting the raw snapshot let an aliased provider, a trailing slash
    or a blank instruction describe the same vector space under a different
    value -- which the manager then reads as drift and answers with a demand
    for an all-worker restart that changes nothing.
    """
    canonical = _admit(tmp_path).vector_space_fingerprint
    identity = canonical_embedding_identity(authority_embedding_config(_snapshot()))

    assert canonical == embedding_identity_fingerprint(identity)
    for spelling in (
        replace(_snapshot(), endpoint="https://api.openai.com/v1/embeddings/"),
        replace(_snapshot(), provider="openai_embedding"),
        replace(_snapshot(), instruct="   "),
    ):
        assert _admit(tmp_path, spelling).vector_space_fingerprint == canonical


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param({}, {"instruct": "   "}, id="omitted_then_whitespace"),
        pytest.param({"instruct": "   "}, {}, id="whitespace_then_omitted"),
        pytest.param({"instruct": None}, {"instruct": ""}, id="null_then_empty"),
        pytest.param({"instruct": ""}, {"instruct": "\t \n"}, id="empty_then_blanks"),
    ],
)
def test_equivalent_instruct_spellings_are_never_drift(
    monkeypatch, tmp_path, authority_database, first, second
):
    """Omitted, null, empty and whitespace-only instructions are one identity.

    Driven through the real service and the real row, because this is a
    persistence contract: the two spellings have to reach the same stored
    value, so the manager cannot see a second vector space where an
    administrator only re-saved the same one.
    """
    sessions = authority_database
    monkeypatch.setattr(
        manager_module, "_read_authority_snapshot", lambda: _loaded_snapshot(sessions)
    )

    _store_authority(sessions, **first)
    assert _persisted_instruct(sessions) is None
    fingerprint = _loaded_snapshot(sessions).vector_space_fingerprint()

    manager = _manager(monkeypatch, tmp_path / "memory")
    assert manager.admit().state is MemoryLifecycleState.READY
    store = manager.get_memory_store()

    _store_authority(sessions, **second)

    assert _persisted_instruct(sessions) is None
    assert _loaded_snapshot(sessions).vector_space_fingerprint() == fingerprint
    assert manager.check_embedding_model_change() is False
    assert manager.status().state is MemoryLifecycleState.READY
    assert manager.get_memory_store() is store


@pytest.mark.parametrize("stored", ["", "   ", "\t\n"])
def test_a_blank_instruct_row_materializes_as_the_omitted_identity(
    authority_database, stored
):
    """A row an earlier build persisted blank still names the same space."""
    sessions = authority_database
    _store_authority(sessions)
    omitted = _loaded_snapshot(sessions)

    _write_raw_instruct(sessions, stored)
    blank = _loaded_snapshot(sessions)

    assert blank.instruct is None
    assert blank.vector_space_fingerprint() == omitted.vector_space_fingerprint()
    assert blank.authority_fingerprint() == omitted.authority_fingerprint()
    with sessions() as db:
        record = GlobalMemoryEmbeddingAuthorityService(db).read_record()
    assert record is not None and record.instruct is None


# --------------------------------------------------------------------------
# Drift: meaningful changes only, and never a stale adapter.
# --------------------------------------------------------------------------


def test_meaningful_drift_revokes_the_publication(monkeypatch, tmp_path):
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    published = manager._publication
    assert published is not None

    # A different embedding model is a different vector space.
    state["snapshot"] = _snapshot(model_name="text-embedding-3-large")

    assert manager.check_embedding_model_change() is True
    assert manager._publication is None
    with pytest.raises(MemoryUnavailableError) as raised:
        manager.get_memory_store()
    assert raised.value.status.state is MemoryLifecycleState.RESTART_REQUIRED


def test_drift_is_detected_on_the_read_path_not_only_on_demand(monkeypatch, tmp_path):
    """Staleness is structural: the store cannot be handed out after drift."""
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    assert manager.get_memory_store() is not None

    state["snapshot"] = _snapshot(dimension=8)

    with pytest.raises(MemoryUnavailableError):
        manager.get_memory_store()


def test_deleting_the_authority_is_drift(monkeypatch, tmp_path):
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY

    state["snapshot"] = None

    with pytest.raises(MemoryUnavailableError) as raised:
        manager.get_memory_store()
    assert raised.value.status.state is MemoryLifecycleState.RESTART_REQUIRED


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(
            lambda: _snapshot(api_key="rotated", credential_identity="identity-2"),
            id="credential_rotation",
        ),
        pytest.param(lambda: _snapshot(max_retries=9), id="retry_budget"),
    ],
)
def test_non_vector_space_changes_are_not_drift(monkeypatch, tmp_path, replacement):
    """Timestamps, provenance and credentials never move the vector space."""
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    store = manager.get_memory_store()

    replaced = replacement()
    assert replaced.authority_fingerprint() != _snapshot().authority_fingerprint()
    assert replaced.vector_space_fingerprint() == _snapshot().vector_space_fingerprint()
    state["snapshot"] = replaced

    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is store


def test_equivalent_row_replacement_is_not_drift(monkeypatch, tmp_path):
    """Re-writing the same authority moves only its timestamps."""
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    store = manager.get_memory_store()

    later = datetime.now(timezone.utc)
    _install_authority(
        monkeypatch,
        GlobalMemoryEmbeddingAuthoritySnapshot(
            **{
                **{
                    field: getattr(_snapshot(), field)
                    for field in (
                        "provider",
                        "model_name",
                        "endpoint",
                        "dimension",
                        "instruct",
                        "max_retries",
                        "credential_source",
                        "global_sharing_consent",
                        "consented_by_actor_subject",
                        "credential_status",
                        "api_key",
                        "credential_identity",
                    )
                },
                "consented_at": later,
                "created_at": later,
                "updated_at": later,
            }
        ),
    )

    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is store


def test_transient_database_failure_is_not_drift(monkeypatch, tmp_path):
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    store = manager.get_memory_store()

    state["error"] = AuthorityUnreadable("database is briefly unavailable")

    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is store
    assert manager.status().state is MemoryLifecycleState.READY


def test_credential_failure_after_admission_is_not_drift(monkeypatch, tmp_path):
    """A broken credential does not change what the stored vectors mean."""
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    store = manager.get_memory_store()

    state["error"] = AuthorityCredentialUnavailable("credential unavailable")

    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is store


@pytest.mark.asyncio
async def test_drift_check_pool_timeout_runs_off_loop_and_stays_loud(
    monkeypatch, tmp_path
):
    """A real QueuePool wait must run off-loop and remain a visible failure.

    The drift check is the only remaining path that reads the authority for a
    caller, so it is where pool exhaustion can still reach one. Two things have
    to hold there: the synchronous checkout must happen on a worker thread
    rather than on the event loop, and the timeout must propagate instead of
    being folded into a quiet memory outage.
    """
    read_authority_snapshot = manager_module._read_authority_snapshot
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path / "memory")
    assert manager.admit().state is MemoryLifecycleState.READY

    # Admission is over. From here the drift check must reach the database,
    # which is what makes the exhausted pool below observable at all.
    monkeypatch.setattr(
        manager_module, "_read_authority_snapshot", read_authority_snapshot
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'memory-policy-timeout.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    session_factory = sessionmaker(bind=engine)

    def get_test_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(manager_module, "get_db", get_test_db)
    monkeypatch.setattr(
        agent_service_manager, "get_memory_store", manager.get_memory_store
    )

    held_connection = engine.connect()
    try:
        with assert_pool_checkout_off_loop(engine):
            with pytest.raises(SQLAlchemyTimeoutError):
                await agent_service_manager.resolve_agent_service_memory_policy_async(
                    agent_config={},
                )
    finally:
        held_connection.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_store_info_answers_two_hundred_while_the_pool_is_exhausted(
    monkeypatch, tmp_path
):
    """``/api/memory/store-info`` reports in every state, this one included.

    The report reaches the manager's drift check, which checks out its own
    authority Session while ``get_current_user``'s request-scoped connection
    is still held. On a one-slot, no-overflow pool that nested checkout has
    nowhere to go. Two things have to hold for the route to keep its promise:
    the checkout must run off the event loop, and the boundary must degrade to
    the last published state instead of letting the timeout become a 500.

    Driven through the real router and real Bearer authentication, because a
    fake manager never reaches this boundary at all.
    """
    from httpx import ASGITransport, AsyncClient

    from xagent.web.api import memory as memory_api
    from xagent.web.api.auth import create_access_token, hash_password
    from xagent.web.api.memory import MemoryManagementRouter
    from xagent.web.models.database import Base, get_db
    from xagent.web.models.user import User

    read_authority_snapshot = manager_module._read_authority_snapshot
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path / "memory")
    assert manager.admit().state is MemoryLifecycleState.READY

    # Admission is over. From here the report must reach the database, which
    # is what makes the exhausted pool observable from the route at all.
    monkeypatch.setattr(
        manager_module, "_read_authority_snapshot", read_authority_snapshot
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'store-info-pool.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=EXHAUSTION_POOL_TIMEOUT,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    def get_test_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(manager_module, "get_db", get_test_db)

    with session_factory() as db:
        operator = User(
            username="memory-store-info-operator",
            password_hash=hash_password("operator"),
            is_admin=True,
        )
        db.add(operator)
        db.commit()
        token = create_access_token({"sub": operator.username, "user_id": operator.id})

    app = FastAPI()
    app.include_router(MemoryManagementRouter().get_router())
    app.dependency_overrides[get_db] = get_test_db
    monkeypatch.setattr(memory_api, "get_memory_store_manager", lambda: manager)

    with assert_pool_checkout_off_loop(engine):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get(
                "/api/memory/store-info",
                headers={"Authorization": f"Bearer {token}"},
            )

    assert response.status_code == 200
    body = response.json()
    # The last published state, reported as it stands -- not a fabricated
    # outage, and not the pool timeout leaking into a lifecycle state.
    assert body["state"] == MemoryLifecycleState.READY.value
    assert body["detail"] == memory_lifecycle.PUBLIC_DETAILS[MemoryLifecycleState.READY]
    assert body["is_lancedb"] is True
    assert body["store_type"] == LanceDBMemoryStore.__name__
    assert body["mode"] == MemoryStorageMode.VECTOR.value
    assert body["supports_vector_search"] is True
    engine.dispose()


# --------------------------------------------------------------------------
# Reconciling a cached agent at the turn boundary.
# --------------------------------------------------------------------------


def _cached_agent(store, *, memory_enabled=True, **overrides):
    """A cached AgentService as the manager's cache-hit path would find one."""
    from xagent.core.agent.service import AgentService

    return AgentService(
        name="cached-agent",
        id="cached-agent",
        tools=[],
        memory=store,
        memory_enabled=memory_enabled,
        enable_workspace=False,
        **overrides,
    )


def _cache(agent, *, task_id=7, owner_id=3):
    """An AgentServiceManager already holding ``agent`` for ``task_id``."""
    from xagent.core.execution_scope import scope_fingerprint

    agents = agent_service_manager.AgentServiceManager()
    agents._agents[task_id] = agent
    agents._agent_owner_ids[task_id] = owner_id
    agents._agent_scope_fingerprints[task_id] = scope_fingerprint(None)
    return agents


async def _turn(agents, *, task_id=7, owner_id=3):
    """One cache-hit turn through the public entry point."""
    return await agents.get_agent_for_task(
        task_id,
        task_owner_user_id=owner_id,
        resolved_execution_scope=None,
    )


def _ready_runtime(monkeypatch, tmp_path):
    """An admitted manager wired in as the runtime's memory source."""
    state = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    monkeypatch.setattr(
        agent_service_manager, "get_memory_store", manager.get_memory_store
    )
    return state, manager


@pytest.mark.asyncio
async def test_a_cache_hit_after_drift_runs_with_memory_disabled(monkeypatch, tmp_path):
    """The cache-hit path must not hand out a service built for a dead space.

    The hit path re-checks owner and scope invariants only, so a service built
    while memory was serving kept ``memory_enabled``, the published store and
    its execution-scoped memory tools after an administrator changed the
    authority's vector space -- reading and writing through the adapter built
    for the previous one.
    """
    state, manager = _ready_runtime(monkeypatch, tmp_path)
    published = manager.get_memory_store()
    agent = _cached_agent(published)
    # Built before the drift, exactly as a live turn would have left it.
    agent._execution_adapter = agent._build_execution_adapter()
    assert agent._execution_adapter.config.memory_store is published
    agents = _cache(agent)

    # An administrator repoints the authority between turns.
    state["snapshot"] = _snapshot(model_name="text-embedding-3-large")

    returned = await _turn(agents)

    assert returned is agent
    assert agent.memory_enabled is False
    assert agent.memory_available is False
    reason = MemoryLifecycleState.RESTART_REQUIRED.value
    assert agent.memory_availability_reason == reason
    status = agent.get_status()
    assert status["memory_available"] is False
    assert status["memory_availability_reason"] == reason
    assert agent.execution_metadata["memory_available"] is False
    assert agent.execution_metadata["memory_availability_reason"] == reason


@pytest.mark.asyncio
async def test_a_reconciled_service_cannot_reach_the_old_store(monkeypatch, tmp_path):
    """Neither through the service, nor through the tools built from it."""
    state, manager = _ready_runtime(monkeypatch, tmp_path)
    published = manager.get_memory_store()
    agent = _cached_agent(published)
    agent._execution_adapter = agent._build_execution_adapter()
    agents = _cache(agent)

    state["snapshot"] = _snapshot(model_name="text-embedding-3-large")
    await _turn(agents)

    assert agent.memory is not published
    assert isinstance(unwrap_memory_store(agent.memory), InMemoryMemoryStore)
    # The compatibility shim no longer points at the revoked proxy either.
    assert agent.agent.memory_store is agent.memory
    # No store reaches the runtime at all, so no memory tool is built from one.
    assert agent._execution_adapter.config.memory_store is None
    assert (
        agent._execution_adapter.config.execution_metadata["memory_availability_reason"]
        == MemoryLifecycleState.RESTART_REQUIRED.value
    )
    # And the reference anyone still holds fails closed on its own check.
    with pytest.raises(MemoryUnavailableError):
        published.add(MemoryNote(content="after the change"))


@pytest.mark.asyncio
async def test_a_ready_cache_hit_costs_one_authority_read_and_changes_nothing(
    monkeypatch, tmp_path
):
    """The reconciliation is the drift check, not an extra read on top of it.

    It must also stay off every per-operation path: the proxy's own liveness
    check is lock-free by contract and never reads the authority, so a turn
    that does any amount of memory work still costs this one read.
    """
    state, manager = _ready_runtime(monkeypatch, tmp_path)
    published = manager.get_memory_store()
    agent = _cached_agent(published)
    agents = _cache(agent)

    before = state["reads"]
    returned = await _turn(agents)
    reads_for_the_turn = state["reads"] - before

    assert reads_for_the_turn == 1
    assert returned is agent
    assert agent.memory is published
    assert agent.memory_enabled is True
    assert agent.memory_available is True
    assert agent.memory_availability_reason is None

    with UserContext(11):
        published.add(MemoryNote(content="ordinary memory traffic"))
        assert published.list_all()
    assert state["reads"] - before == reads_for_the_turn


@pytest.mark.asyncio
async def test_a_service_already_without_memory_reads_nothing_and_stays_off(
    monkeypatch, tmp_path
):
    """One-directional: this path disables, it never re-enables.

    A preview or an agent-backed task runs with memory off by configuration,
    not by lifecycle, and the cache-hit path does not carry the inputs that
    decided that -- so a healthy lifecycle must not turn memory back on.
    """
    state, _manager_ = _ready_runtime(monkeypatch, tmp_path)
    agent = _cached_agent(
        InMemoryMemoryStore(),
        memory_enabled=False,
        memory_available=False,
        memory_availability_reason="blocked_repair",
    )
    agents = _cache(agent)

    before = state["reads"]
    await _turn(agents)

    assert state["reads"] == before
    assert agent.memory_enabled is False
    assert agent.memory_available is False
    assert agent.memory_availability_reason == "blocked_repair"


# --------------------------------------------------------------------------
# Failure semantics.
# --------------------------------------------------------------------------


def test_credential_failure_at_admission_fails_closed(monkeypatch, tmp_path):
    _install_authority(
        monkeypatch, None, error=AuthorityCredentialUnavailable("unusable")
    )
    manager = _manager(monkeypatch, tmp_path)

    status = manager.admit()

    assert status.state is MemoryLifecycleState.CREDENTIAL_UNAVAILABLE
    assert manager._publication is None
    with pytest.raises(MemoryUnavailableError):
        manager.get_memory_store()


def test_transient_failure_at_admission_stays_fenced_until_restart(
    monkeypatch, tmp_path
):
    """A retryably-failed worker never re-admits online, even once healthy.

    Admission repairs by rewriting the table, and ``writers_quiesced`` is only
    a claim about this process. Re-attempting after the fleet is live could
    destroy another worker's commits, so recovery is an operator restart.
    """
    state = _install_authority(
        monkeypatch, None, error=AuthorityUnreadable("transient")
    )
    manager = _manager(monkeypatch, tmp_path)

    assert manager.admit().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    assert manager._publication is None

    # The underlying fault clears; the fenced worker must not notice.
    state["error"] = None
    state["snapshot"] = _snapshot()

    assert manager.status().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    with pytest.raises(MemoryUnavailableError):
        manager.get_memory_store()

    # Only a restart -- a fresh manager running its own startup -- recovers.
    restarted = _manager(monkeypatch, tmp_path)
    assert restarted.admit().state is MemoryLifecycleState.READY


def test_request_and_status_paths_never_admit(monkeypatch, tmp_path):
    """No request or status path may reach admission's table-rewriting repair."""
    state = _install_authority(monkeypatch, _snapshot())
    admissions: list[int] = []

    real = memory_lifecycle.admit_authority_storage

    def counting_admit(snapshot, **kwargs):
        admissions.append(1)
        kwargs["db_dir"] = str(tmp_path)
        kwargs["embedding_factory"] = lambda _config: ConstantEmbedding()
        return real(snapshot, **kwargs)

    monkeypatch.setattr(manager_module, "admit_authority_storage", counting_admit)
    manager = DynamicMemoryStoreManager()

    # Startup admission has not run. Every caller-facing path must fail closed
    # without admitting, however many times it is called.
    for _ in range(5):
        assert manager.acquire() == (
            None,
            manager._status,
        )
        assert manager.status().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
        assert (
            manager.get_store_info()["state"]
            is MemoryLifecycleState.RETRYABLE_UNAVAILABLE.value
        )
        with pytest.raises(MemoryUnavailableError):
            manager.get_memory_store()

    assert admissions == []
    # The authority row was never read either: a status call is not a probe.
    assert state["reads"] == 0

    # admit() is the one door, and it opens exactly once.
    assert manager.admit().state is MemoryLifecycleState.READY
    assert len(admissions) == 1
    manager.status()
    manager.get_store_info()
    assert len(admissions) == 1


def test_a_retryably_fenced_manager_reports_retryable_unavailable(
    monkeypatch, tmp_path
):
    """The fenced state an operator restarts on stays visible on every path."""
    _install_authority(monkeypatch, None, error=AuthorityUnreadable("down"))
    manager = _manager(monkeypatch, tmp_path)

    assert manager.admit().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE

    for _ in range(3):
        assert manager.status().state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE
        info = manager.get_store_info()
        assert info["state"] == MemoryLifecycleState.RETRYABLE_UNAVAILABLE.value
        assert info["store_type"] is None
        assert info["is_lancedb"] is False
        with pytest.raises(MemoryUnavailableError) as raised:
            manager.get_memory_store()
        assert raised.value.status.state is MemoryLifecycleState.RETRYABLE_UNAVAILABLE


@pytest.mark.parametrize(
    "state",
    [
        MemoryLifecycleState.BLOCKED_REPAIR,
        MemoryLifecycleState.CREDENTIAL_UNAVAILABLE,
        MemoryLifecycleState.RESTART_REQUIRED,
        MemoryLifecycleState.NOT_CONFIGURED,
    ],
)
def test_terminal_states_are_never_re_admitted_online(monkeypatch, tmp_path, state):
    holder = _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    manager._status = memory_lifecycle.MemoryLifecycleStatus(state)
    before = holder["reads"]

    manager.admit()

    assert holder["reads"] == before
    assert manager._status.state is state


def test_failed_re_admission_leaves_the_manager_untouched(monkeypatch, tmp_path):
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    published = manager._publication
    status = manager._status

    # Model an unsettled manager that is nonetheless already serving, then let
    # the next admission attempt fail outright.
    manager._status = memory_lifecycle.MemoryLifecycleStatus(
        MemoryLifecycleState.RETRYABLE_UNAVAILABLE
    )
    monkeypatch.setattr(
        manager_module,
        "admit_authority_storage",
        lambda *_args, **_kwargs: memory_lifecycle.AdmissionResult(
            memory_lifecycle.MemoryLifecycleStatus(
                MemoryLifecycleState.RETRYABLE_UNAVAILABLE
            )
        ),
    )

    manager.admit()

    assert manager._publication is published
    assert manager._status == status


def test_no_authority_serves_an_ephemeral_store(monkeypatch, tmp_path):
    _install_authority(monkeypatch, None)
    manager = _manager(monkeypatch, tmp_path)

    status = manager.admit()

    assert status.state is MemoryLifecycleState.NOT_CONFIGURED
    assert status.vector_search is False
    store = manager.get_memory_store()
    # Published through the revocation wrapper, with user isolation intact
    # underneath it and the ephemeral store at the bottom.
    assert isinstance(store, RevocableMemoryStore)
    assert isinstance(store._base_store, UserIsolatedMemoryStore)
    assert isinstance(unwrap_memory_store(store), InMemoryMemoryStore)


def test_reinitialization_is_refused(monkeypatch, tmp_path, caplog):
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    assert manager.admit().state is MemoryLifecycleState.READY
    published = manager._publication

    with caplog.at_level("WARNING"):
        manager.force_reinitialize()

    assert manager._publication is published
    assert "all-worker restart" in caplog.text


def test_store_info_is_public_safe(monkeypatch, tmp_path):
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)
    manager.admit()

    info = manager.get_store_info()

    assert info["state"] == "ready"
    assert info["is_lancedb"] is True
    assert info["mode"] == "vector"
    assert info["supports_vector_search"] is True
    assert info["store_type"] == LanceDBMemoryStore.__name__
    serialized = json.dumps(info)
    assert "secret-key" not in serialized
    assert str(tmp_path) not in serialized


def test_operator_guidance_covers_every_non_ready_state():
    for state in MemoryLifecycleState:
        status = memory_lifecycle.MemoryLifecycleStatus(state)
        assert status.detail  # every state has caller-safe wording
        if state is not MemoryLifecycleState.READY:
            guidance = memory_lifecycle.OPERATOR_GUIDANCE[state]
            assert "restart" in guidance.lower()


def test_every_fenced_state_is_indistinguishable_to_callers():
    """Public detail never tells an unprivileged caller which fault it hit.

    Asserted over the whole matrix rather than a chosen few, because the API
    returns ``status.detail`` verbatim: one fenced state wording itself
    differently -- ``retryable_unavailable`` used to say "temporarily
    unavailable" -- is enough to let a caller separate a transient backend
    failure from a credential fault or from invalid legacy data.
    """
    fenced = set(MemoryLifecycleState) - {
        MemoryLifecycleState.READY,
        MemoryLifecycleState.NOT_CONFIGURED,
    }
    # A state added later is fenced unless it is deliberately classified as
    # servable, so this matrix cannot silently stop covering one.
    assert memory_lifecycle.FENCED_STATES == fenced

    details = {memory_lifecycle.MemoryLifecycleStatus(state).detail for state in fenced}
    assert details == {memory_lifecycle.FENCED_DETAIL}
    # The distinguishing text still exists -- for the operator log only.
    guidance = {memory_lifecycle.OPERATOR_GUIDANCE[state] for state in fenced}
    assert len(guidance) == len(fenced)


# --------------------------------------------------------------------------
# Multi-worker startup and a full read/write cycle.
# --------------------------------------------------------------------------


def _worker_admit(database_path, results):
    from xagent.providers.vector_store.lancedb import (
        clear_connection_cache as clear_cache,
    )

    clear_cache()
    results.put(
        admit_authority_storage(
            _snapshot(),
            db_dir=str(database_path),
            embedding_factory=lambda _config: ConstantEmbedding(),
        ).status.state.value
    )


def test_concurrent_worker_startup_admits_one_consistent_table(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_worker_admit, args=(tmp_path, results))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
        assert worker.exitcode == 0

    states = sorted(results.get(timeout=30) for _ in workers)
    assert set(states) <= {"ready", "retryable_unavailable"}

    clear_connection_cache()
    connection = lancedb.connect(tmp_path)
    assert list(connection.table_names()) == [MEMORY_TABLE_NAME]
    # Whatever the interleaving, a later worker admits the one table cleanly.
    assert _admit(tmp_path).status.state is MemoryLifecycleState.READY


def test_startup_admission_is_serialized_within_a_worker(monkeypatch, tmp_path):
    """Concurrent startup admissions admit once and publish one store."""
    _install_authority(monkeypatch, _snapshot())
    manager = _manager(monkeypatch, tmp_path)

    stores: list[Any] = []
    barrier = threading.Barrier(4)

    def acquire():
        barrier.wait()
        manager.admit()
        stores.append(manager.get_memory_store())

    threads = [threading.Thread(target=acquire) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert len(stores) == 4
    assert all(store is stores[0] for store in stores)


def test_admitted_store_round_trips_reads_and_writes(tmp_path):
    result = _admit(tmp_path)
    store = result.store
    assert store is not None

    with UserContext(4321):
        added = store.add(MemoryNote(content="the kettle is in the third cupboard"))
        assert added.success
        note_id = added.memory_id

        fetched = store.get(note_id)
        assert fetched.success
        assert fetched.content.content == "the kettle is in the third cupboard"

        assert [note.id for note in store.list_all(None)] == [note_id]
        assert [note.id for note in store.search("kettle", k=5)] == [note_id]

        assert store.delete(note_id).success
        assert store.list_all(None) == []


def test_admitted_store_isolates_users(tmp_path):
    store = _admit(tmp_path).store
    assert store is not None

    with UserContext(11):
        mine = store.add(MemoryNote(content="user eleven note"))
        assert mine.success
    with UserContext(12):
        assert store.list_all(None) == []
        assert store.get(mine.memory_id).success is False


def test_second_admission_of_a_populated_table_is_ready(tmp_path):
    store = _admit(tmp_path).store
    assert store is not None
    with UserContext(7):
        assert store.add(MemoryNote(content="survives a restart")).success

    clear_connection_cache()
    second = _admit(tmp_path)

    assert second.status.state is MemoryLifecycleState.READY
    assert second.store is not None
    with UserContext(7):
        assert [note.content for note in second.store.list_all(None)] == [
            "survives a restart"
        ]


# --------------------------------------------------------------------------
# TEXT_ONLY admission never receives the authority's embedding adapter.
#
# A table whose stored vectors were written under a different identity is
# admitted TEXT_ONLY: writable, no vector search. Handing it the authority's
# adapter would make the first ordinary write rewrite the table to the new
# width and re-embed every historical row -- the online re-embedding the
# lifecycle forbids. These tests pin that it does not happen.
# --------------------------------------------------------------------------


class _SizedEmbedding(BaseEmbedding):
    """Deterministic embedding of an arbitrary width."""

    def __init__(self, dimension: int, seed: float = 0.25) -> None:
        self._dimension = dimension
        self._seed = seed

    def encode(self, text, dimension=None, instruct=None):
        vector = [self._seed] * self._dimension
        return vector if isinstance(text, str) else [vector for _ in text]

    def get_dimension(self):
        return self._dimension

    @property
    def abilities(self):
        return ["embed"]


def _table_vectors(tmp_path) -> dict[str, Optional[list[float]]]:
    """Read every row's vector straight off disk, bypassing the store."""
    clear_connection_cache()
    connection = lancedb.connect(tmp_path)
    table = connection.open_table(MEMORY_TABLE_NAME)
    try:
        rows = table.to_arrow().to_pylist()
        return {row["id"]: row.get("vector") for row in rows}
    finally:
        _safe_close_table(table)


def _table_vector_dimension(tmp_path) -> Optional[int]:
    """The declared on-disk vector width, or ``None`` when there is no column."""
    clear_connection_cache()
    connection = lancedb.connect(tmp_path)
    table = connection.open_table(MEMORY_TABLE_NAME)
    try:
        if "vector" not in table.schema.names:
            return None
        return int(table.schema.field("vector").type.list_size)
    finally:
        _safe_close_table(table)


def _seed_vector_table(tmp_path, *, user_id: int = 99) -> str:
    """Admit a 4-dimension authority and leave one ordinary note behind."""
    seeded = admit_authority_storage(
        _snapshot(dimension=DIMENSION),
        db_dir=str(tmp_path),
        embedding_factory=lambda _config: _SizedEmbedding(DIMENSION),
    )
    assert seeded.status.mode is MemoryStorageMode.VECTOR
    assert seeded.store is not None
    with UserContext(user_id):
        added = seeded.store.add(MemoryNote(content="the older note about kettles"))
    assert added.success
    clear_connection_cache()
    return added.memory_id


@pytest.fixture
def text_only_admission(tmp_path):
    """A 4-dimension table admitted under an 8-dimension authority."""
    old_note_id = _seed_vector_table(tmp_path)
    assert _table_vector_dimension(tmp_path) == DIMENSION

    wider = _snapshot(model_name="text-embedding-3-large", dimension=8)
    built: list[BaseEmbedding] = []

    def factory(_config):
        adapter = _SizedEmbedding(8)
        built.append(adapter)
        return adapter

    result = admit_authority_storage(
        wider, db_dir=str(tmp_path), embedding_factory=factory
    )
    return result, old_note_id, built


def test_mismatching_vectors_admit_text_only_without_vector_search(
    text_only_admission,
):
    result, _old_note_id, _built = text_only_admission

    assert result.status.state is MemoryLifecycleState.READY
    assert result.status.mode is MemoryStorageMode.TEXT_ONLY
    assert result.status.vector_search is False
    assert result.store is not None


def test_text_only_admission_never_builds_the_authority_adapter(text_only_admission):
    """The adapter is not merely withheld from the store; it is never built."""
    _result, _old_note_id, built = text_only_admission

    assert built == []


def test_text_only_write_does_not_re_embed_or_widen_the_table(
    tmp_path, text_only_admission
):
    """One ordinary write must not rewrite the table under the new identity."""
    result, old_note_id, _built = text_only_admission
    store = result.store
    assert store is not None
    before = _table_vectors(tmp_path)

    with UserContext(99):
        added = store.add(MemoryNote(content="a newer note about saucepans"))
    assert added.success

    # (a) the on-disk vector width is untouched
    assert _table_vector_dimension(tmp_path) == DIMENSION

    after = _table_vectors(tmp_path)
    # (b) the pre-existing row's vector is byte-for-byte what it was
    assert after[old_note_id] == before[old_note_id]
    assert after[old_note_id] is not None
    assert len(after[old_note_id]) == DIMENSION
    # (c) the new row carries no vector at all
    assert after[added.memory_id] is None

    # (d) both rows are still reachable, through the lexical fallback
    with UserContext(99):
        found = {note.id for note in store.search("note", k=10)}
    assert found == {old_note_id, added.memory_id}


def test_same_width_identity_drift_also_gets_no_adapter(tmp_path):
    """Drift that keeps the dimension is still a different vector space."""
    old_note_id = _seed_vector_table(tmp_path, user_id=5)
    before = _table_vectors(tmp_path)

    # Same dimension, different model and endpoint: no rewrite would betray
    # this at the schema level, so mixing the two spaces would go unnoticed.
    drifted = _snapshot(model_name="text-embedding-ada-002", dimension=DIMENSION)
    built: list[BaseEmbedding] = []

    def factory(_config):
        adapter = _SizedEmbedding(DIMENSION, seed=0.9)
        built.append(adapter)
        return adapter

    result = admit_authority_storage(
        drifted, db_dir=str(tmp_path), embedding_factory=factory
    )

    assert result.status.mode is MemoryStorageMode.TEXT_ONLY
    assert result.status.vector_search is False
    assert built == []

    store = result.store
    assert store is not None
    with UserContext(5):
        added = store.add(MemoryNote(content="written after the identity moved"))
    assert added.success

    after = _table_vectors(tmp_path)
    assert _table_vector_dimension(tmp_path) == DIMENSION
    assert after[old_note_id] == before[old_note_id]
    assert after[added.memory_id] is None


def test_vector_mode_still_receives_the_authority_adapter(tmp_path):
    """The fix must not starve the mode that is entitled to the adapter."""
    built: list[BaseEmbedding] = []

    def factory(_config):
        adapter = _SizedEmbedding(DIMENSION)
        built.append(adapter)
        return adapter

    result = admit_authority_storage(
        _snapshot(dimension=DIMENSION),
        db_dir=str(tmp_path),
        embedding_factory=factory,
    )

    assert result.status.mode is MemoryStorageMode.VECTOR
    assert len(built) == 1

    store = result.store
    assert store is not None
    with UserContext(3):
        added = store.add(MemoryNote(content="vector mode still embeds"))
    assert added.success
    assert _table_vectors(tmp_path)[added.memory_id] is not None
