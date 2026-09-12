from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from xagent.web.api.admin_memory_embedding_authority import router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.global_memory_embedding_authority import (
    GlobalMemoryEmbeddingAuthority,
)
from xagent.web.services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    AuthorityCredentialUnavailable,
    GlobalMemoryEmbeddingAuthorityService,
)

SECRET = "application-owned-secret"
PAYLOAD = {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "endpoint": "https://api.openai.com/v1",
    "dimension": 1536,
    "max_retries": 10,
    "api_key": SECRET,
}


@pytest.fixture
def authority_harness(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'authority.db'}")
    GlobalMemoryEmbeddingAuthority.__table__.create(engine)
    sessions = sessionmaker(bind=engine)
    actor = SimpleNamespace(is_admin=True, actor_subject="actor-1")
    app = FastAPI()
    app.include_router(router)

    def current_user():
        return actor

    def database():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_current_user] = current_user
    app.dependency_overrides[get_db] = database
    with TestClient(app) as client:
        yield client, sessions, actor, engine


@pytest.mark.parametrize("method", ["get", "put", "delete"])
def test_authority_endpoints_require_admin(authority_harness, method):
    client, _sessions, actor, _engine = authority_harness
    actor.is_admin = False
    response = client.request(
        method.upper(),
        "/api/admin/memory/embedding-authority",
        json=PAYLOAD if method == "put" else None,
    )
    assert response.status_code == 403


def test_authority_crud_is_public_safe_and_ignores_personal_models(authority_harness):
    client, sessions, _actor, engine = authority_harness
    statements = []
    event.listen(
        engine, "before_cursor_execute", lambda *args: statements.append(args[2])
    )
    response = client.put("/api/admin/memory/embedding-authority", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["credential_status"] == "configured"
    assert SECRET not in response.text
    assert "credential_digest" not in response.text
    assert all(
        "user_default_models" not in sql and " models " not in sql for sql in statements
    )
    with sessions() as db:
        row = db.get(GlobalMemoryEmbeddingAuthority, "global")
        assert row.api_key_encrypted != SECRET
    assert client.delete("/api/admin/memory/embedding-authority").status_code == 204


def test_equivalent_updates_keep_one_semantic_identity(authority_harness):
    _client, sessions, _actor, _engine = authority_harness
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="actor-1")
        first = service.load_snapshot()
        equivalent = dict(
            PAYLOAD, provider="openai-compatible", endpoint="https://api.openai.com/v1/"
        )
        service.set(AuthorityConfiguration(**equivalent), actor_subject="actor-2")
        second = service.load_snapshot()
        assert db.query(GlobalMemoryEmbeddingAuthority).count() == 1
        assert first.semantic_fingerprint() == second.semantic_fingerprint()
        assert SECRET not in repr(second)
        changed = dict(PAYLOAD, api_key="replacement-secret")
        service.set(AuthorityConfiguration(**changed), actor_subject="actor-2")
        assert (
            second.semantic_fingerprint()
            != service.load_snapshot().semantic_fingerprint()
        )


def test_malformed_credential_is_safely_classified(authority_harness):
    _client, sessions, _actor, _engine = authority_harness
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="actor-1")
        row = service.get_row()
        row.api_key_encrypted = (
            Fernet(Fernet.generate_key()).encrypt(b"wrong-key").decode()
        )
        db.commit()
        assert service.credential_status() == "unavailable"
        with pytest.raises(AuthorityCredentialUnavailable) as error:
            service.load_snapshot()
        assert str(error.value) == "Global memory embedding credential is unavailable"
        assert SECRET not in repr(row)
