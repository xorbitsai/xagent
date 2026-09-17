import hashlib
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import get_cipher
from xagent.web.api.admin_memory_embedding_authority import router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.global_memory_embedding_authority import (
    GlobalMemoryEmbeddingAuthority,
)
from xagent.web.services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    AuthorityCredentialUnavailable,
    CredentialSource,
    GlobalMemoryEmbeddingAuthorityService,
)

SECRET = "application-owned-secret"
MARKER = "leaked-credential-marker"
URL = "/api/admin/memory/embedding-authority"
TABLE = GlobalMemoryEmbeddingAuthority.__tablename__
PAYLOAD = {
    "provider": "openai",
    "model_name": "text-embedding-3-small",
    "endpoint": "https://api.openai.com/v1",
    "dimension": 1536,
    "max_retries": 10,
    "credential_source": "application_owned",
    "global_sharing_consent": True,
    "api_key": SECRET,
}


def _delete_row_once(sessions, armed):
    """An ``after_commit`` hook racing a delete into the committed write."""

    def hook(_session):
        if armed:
            return
        armed.append(True)
        with sessions() as other:
            other.query(GlobalMemoryEmbeddingAuthority).delete()
            other.commit()

    return hook


def _identities(service):
    snapshot = service.load_snapshot()
    return snapshot.authority_fingerprint(), snapshot.vector_space_fingerprint()


def _row_state(sessions):
    with sessions() as db:
        row = db.get(GlobalMemoryEmbeddingAuthority, "global")
        if row is None:
            return None
        return tuple(getattr(row, column.name) for column in row.__table__.columns)


@pytest.fixture
def authority_harness(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'authority.db'}")
    GlobalMemoryEmbeddingAuthority.__table__.create(engine)
    sessions = sessionmaker(bind=engine)
    actor = SimpleNamespace(is_admin=True, actor_subject="actor-1")
    app = FastAPI()
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def echo(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """Mirror of the shared ``/api`` handler: logs and echoes ``input``."""
        logging.getLogger("xagent.web.app").error("Validation error: %s", exc)
        return JSONResponse(422, jsonable_encoder({"detail": exc.errors()}))

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
    assert "credential_verifier" not in response.text
    assert all(
        "user_default_models" not in sql and " models " not in sql for sql in statements
    )
    with sessions() as db:
        row = db.get(GlobalMemoryEmbeddingAuthority, "global")
        assert row.api_key_encrypted != SECRET
        assert row.credential_verifier.startswith("hmac-sha256$v1$")
        assert row.credential_verifier != hashlib.sha256(SECRET.encode()).hexdigest()
    assert client.delete("/api/admin/memory/embedding-authority").status_code == 204


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@example.com/v1",
        "https://example.com/v1?api_key=test-secret",
        "https://example.com/v1#test-secret",
        "ftp://example.com/v1",
        "/v1",
        "https:///v1",
        "https://bad_host/v1",
        "https://[::1/v1",
        "https://example.com:99999/v1",
        "https://example.com:0/v1",
        "https://exa\nmple.com/v1",
    ],
)
def test_invalid_endpoint_is_rejected_without_changing_state(
    authority_harness, endpoint
):
    client, sessions, _actor, _engine = authority_harness
    assert client.put(URL, json=PAYLOAD).status_code == 200
    before = _row_state(sessions)
    response = client.put(URL, json=dict(PAYLOAD, endpoint=endpoint))
    assert response.status_code == 400
    assert endpoint not in response.text
    assert client.get(URL).json()["endpoint"] == "https://api.openai.com/v1/embeddings"
    assert _row_state(sessions) == before


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
        assert first.authority_fingerprint() == second.authority_fingerprint()
        assert first.vector_space_fingerprint() == second.vector_space_fingerprint()
        assert SECRET not in repr(second)


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


@pytest.mark.parametrize(
    "verifier",
    [
        hashlib.sha256(SECRET.encode()).hexdigest(),
        "hmac-sha256$v2$" + "0" * 64,
        "malformed-☃",
    ],
)
def test_invalid_or_legacy_verifier_fails_closed_and_put_recovers(
    authority_harness, verifier
):
    client, sessions, _actor, _engine = authority_harness
    assert (
        client.put("/api/admin/memory/embedding-authority", json=PAYLOAD).status_code
        == 200
    )
    with sessions() as db:
        row = db.get(GlobalMemoryEmbeddingAuthority, "global")
        row.credential_verifier = verifier
        db.commit()
    state = client.get("/api/admin/memory/embedding-authority").json()
    assert state["credential_status"] == "unavailable"
    recovered = client.put("/api/admin/memory/embedding-authority", json=PAYLOAD)
    assert recovered.json()["credential_status"] == "configured"


def test_unavailable_encryption_key_fails_closed(authority_harness, monkeypatch):
    _client, sessions, _actor, _engine = authority_harness
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="actor-1")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        get_cipher.cache_clear()
        try:
            assert service.credential_status() == "unavailable"
        finally:
            get_cipher.cache_clear()


def test_overlong_credential_bearing_endpoint_is_rejected_without_leaking(
    authority_harness, caplog
):
    """A length bound on the request field would publish the endpoint: the
    shared ``/api`` validation handler echoes the raw ``input`` into the 422
    body and logs it, so the bound belongs to the canonicalizer instead."""
    client, sessions, _actor, _engine = authority_harness
    assert client.put(URL, json=PAYLOAD).status_code == 200
    before = _row_state(sessions)
    marker = "leaked-endpoint-userinfo-marker"
    overlong = f"https://admin:{marker}@{'a' * 520}.example.com/v1"
    with caplog.at_level(logging.DEBUG):
        response = client.put(URL, json=dict(PAYLOAD, endpoint=overlong))
    assert len(overlong) > 500
    assert response.status_code == 400
    assert marker not in response.text and marker not in caplog.text
    assert _row_state(sessions) == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"credential_source": None},
        {"credential_source": "personal"},
        {"credential_source": "unknown"},
        {"global_sharing_consent": None},
        {"global_sharing_consent": False},
    ],
)
def test_unowned_or_unconsented_credential_is_rejected(authority_harness, overrides):
    """Ownership and consent are explicit facts, never inferred from admin rights."""
    client, sessions, _actor, _engine = authority_harness
    payload = {k: v for k, v in dict(PAYLOAD, **overrides).items() if v is not None}
    response = client.put(URL, json=payload)
    assert response.status_code == 400
    assert SECRET not in response.text
    assert client.get(URL).json()["configured"] is False
    assert _row_state(sessions) is None


def test_consent_provenance_is_server_side_and_outside_the_fingerprint(
    authority_harness,
):
    client, sessions, _actor, _engine = authority_harness
    before = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    body = client.put(
        URL, json=dict(PAYLOAD, consented_by_actor_subject="spoofed-actor")
    ).json()
    assert body["credential_source"] == "application_owned"
    assert body["global_sharing_consent"] is True
    assert "consented_by_actor_subject" not in body
    assert "actor-1" not in str(body) and "actor-1" not in client.get(URL).text
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        first = service.load_snapshot()
        assert first.credential_source is CredentialSource.APPLICATION_OWNED
        assert first.global_sharing_consent is True
        assert first.consented_by_actor_subject == "actor-1"
        assert first.consented_at.replace(tzinfo=None) >= before
        service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="actor-2")
        second = service.load_snapshot()
        assert second.consented_by_actor_subject == "actor-2"
        assert first.authority_fingerprint() == second.authority_fingerprint()
        assert first.vector_space_fingerprint() == second.vector_space_fingerprint()


@pytest.mark.parametrize(
    "assignment", ["credential_source = 'personal'", "global_sharing_consent = 0"]
)
def test_ownership_and_consent_are_enforced_at_rest(authority_harness, assignment):
    """A row written around the service cannot claim the global authority."""
    client, sessions, _actor, _engine = authority_harness
    assert client.put(URL, json=PAYLOAD).status_code == 200
    with sessions() as db, pytest.raises(IntegrityError):
        db.execute(text(f"UPDATE {TABLE} SET {assignment}"))
        db.commit()


# Scheme/host case, an explicit default port, a terminal DNS dot, a trailing
# slash and Unicode-vs-IDNA spellings all name the same endpoint.
@pytest.mark.parametrize(
    "provider, spellings, expected",
    [
        (
            "openai",
            "https://api.openai.com/v1 https://API.OpenAI.COM/v1"
            " https://api.openai.com.:443/v1 HTTPS://api.openai.com:443/v1/",
            "https://api.openai.com/v1/embeddings",
        ),
        (
            "xinference",
            "http://bücher.example/v1 http://xn--bcher-kva.example/v1"
            " http://BÜCHER.Example.:80/v1/",
            "http://xn--bcher-kva.example/v1",
        ),
        (
            "xinference",
            "http://[0:0:0:0:0:0:0:1]:9997/v1 http://[::1]:9997/v1/",
            "http://[::1]:9997/v1",
        ),
        (
            "xinference",
            "http://localhost:9997/v1/ http://LocalHost:9997/v1",
            "http://localhost:9997/v1",
        ),
    ],
)
def test_equivalent_endpoint_spellings_share_one_identity(
    authority_harness, provider, spellings, expected
):
    _client, sessions, _actor, _engine = authority_harness
    fingerprints = set()
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        for spelling in spellings.split():
            config = dict(PAYLOAD, provider=provider, endpoint=spelling)
            record = service.set(AuthorityConfiguration(**config), actor_subject="a")
            assert record.endpoint == expected
            assert service.get_row().base_url == expected
            fingerprints.add(_identities(service))
        # A genuinely different host, port or path is a different identity.
        for other in ("http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1", "http://a.io"):
            config = dict(PAYLOAD, provider="xinference", endpoint=other)
            service.set(AuthorityConfiguration(**config), actor_subject="a")
            fingerprints.add(_identities(service))
    assert len(fingerprints) == 4


def test_put_returns_written_state_when_a_delete_races_the_commit(authority_harness):
    """A delete landing right after COMMIT must not fail a durable write."""
    client, sessions, _actor, _engine = authority_harness
    hook = _delete_row_once(sessions, [])
    event.listen(Session, "after_commit", hook)
    try:
        response = client.put(URL, json=PAYLOAD)
    finally:
        event.remove(Session, "after_commit", hook)
    body = response.json()
    assert response.status_code == 200
    assert body["configured"] is True
    assert body["endpoint"] == "https://api.openai.com/v1/embeddings"
    assert body["credential_status"] == "configured"
    assert SECRET not in response.text
    assert client.get(URL).json()["configured"] is False


def test_service_set_returns_written_state_when_a_delete_races_the_commit(
    authority_harness,
):
    _client, sessions, _actor, _engine = authority_harness
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        hook = _delete_row_once(sessions, [])
        event.listen(db, "after_commit", hook)
        try:
            record = service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="a1")
        finally:
            event.remove(db, "after_commit", hook)
        assert record.endpoint == "https://api.openai.com/v1/embeddings"
        assert record.credential_source is CredentialSource.APPLICATION_OWNED
        assert record.consented_by_actor_subject == "a1"
        assert service.get_row() is None


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(dict(PAYLOAD, api_key=[MARKER])).encode(),
        json.dumps(dict(PAYLOAD, api_key={"value": MARKER})).encode(),
        b'{"api_key": "' + MARKER.encode() + b'", "provider": "openai",',
    ],
    ids=["api_key-list", "api_key-object", "malformed-json"],
)
def test_credential_bearing_body_never_reaches_the_shared_handler(
    authority_harness, caplog, body
):
    """The route parses the body itself, so no framework 422 can echo it: the
    shared ``/api`` handler reports each error's raw ``input`` and logs the
    exception, and for a non-string credential that input is the secret."""
    client, sessions, _actor, _engine = authority_harness
    assert client.put(URL, json=PAYLOAD).status_code == 200
    before = _row_state(sessions)
    with caplog.at_level(logging.DEBUG):
        response = client.put(
            URL, content=body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert MARKER not in response.text and MARKER not in caplog.text
    assert _row_state(sessions) == before


def test_endpoint_overflowing_after_the_openai_rewrite_is_rejected(authority_harness):
    """``base_url`` is ``String(500)`` and ``/v1`` is rewritten after parsing."""
    client, sessions, _actor, _engine = authority_harness
    overflows = "https://example.com/" + "p" * 477 + "/v1"
    assert len(overflows) == 500
    assert client.put(URL, json=dict(PAYLOAD, endpoint=overflows)).status_code == 400
    assert _row_state(sessions) is None
    fits = "https://example.com/" + "p" * 457 + "/v1"
    stored = client.put(URL, json=dict(PAYLOAD, endpoint=fits)).json()["endpoint"]
    assert stored == fits + "/embeddings" and len(stored) <= 500


@pytest.mark.parametrize(
    "overrides, vector_space_moves",
    [
        ({"api_key": "rotated-secret"}, False),
        ({"max_retries": 3}, False),
        ({"dimension": 512}, True),
    ],
)
def test_authority_identity_covers_more_than_the_vector_space(
    authority_harness, overrides, vector_space_moves
):
    """Key rotation and retry changes move the authority identity only."""
    _client, sessions, _actor, _engine = authority_harness
    with sessions() as db:
        service = GlobalMemoryEmbeddingAuthorityService(db)
        service.set(AuthorityConfiguration(**PAYLOAD), actor_subject="a")
        before = service.load_snapshot()
        service.set(
            AuthorityConfiguration(**dict(PAYLOAD, **overrides)), actor_subject="a"
        )
        after = service.load_snapshot()
        assert after.authority_fingerprint() != before.authority_fingerprint()
        assert (
            after.vector_space_fingerprint() != before.vector_space_fingerprint()
        ) is vector_space_moves
