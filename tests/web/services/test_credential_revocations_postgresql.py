"""The credential revocation outbox on PostgreSQL.

PostgreSQL runs foreign-key cascades and row locks differently from SQLite,
which is exactly where the outbox's guarantees are subtle:

* a cascade from an OAuth client deletes the client row before the cascaded
  grant deletes fire their triggers, so the record would lose the client's
  fields without the ``BEFORE DELETE`` trigger on the client;
* a delete racing a refresh waits for the refresh's row lock and then deletes
  the refreshed row, so the record must hold the rotated tokens;
* replicas claim records with ``FOR UPDATE SKIP LOCKED`` and never two of
  one grant at once.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.utils.encryption import decrypt_value_strict, encrypt_value
from xagent.web.models.credential_revocation import CredentialRevocation
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import credential_revocations as outbox

pytestmark = pytest.mark.postgresql

REVOCATION_ENDPOINT = "https://auth.example.com/revoke"


@pytest.fixture()
def factory():
    with disposable_database_factory("xagent_credential_revocations") as make:
        engine = make("outbox")
        Base.metadata.create_all(
            engine,
            tables=[
                User.__table__,
                UserOAuth.__table__,
                MCPServer.__table__,
                MCPOAuthClient.__table__,
                MCPOAuthGrant.__table__,
                CredentialRevocation.__table__,
            ],
        )
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _seed(factory, *, grants: int = 1) -> dict[str, int]:
    with factory() as db:
        user = User(username="alice", password_hash="x", is_admin=False)
        server = MCPServer.from_config(
            {
                "name": "records",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.example.com/mcp",
                "auth": {"type": "mcp_oauth"},
            }
        )
        db.add_all([user, server])
        db.flush()
        client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer="https://auth.example.com",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="mcp-client",
            client_secret=encrypt_value("mcp-secret"),
            token_endpoint_auth_method="client_secret_basic",
            redirect_uri="https://app.example.com/api/mcp/oauth/callback",
            metadata_json={"revocation_endpoint": REVOCATION_ENDPOINT},
        )
        db.add(client)
        db.flush()
        for index in range(grants):
            db.add(
                MCPOAuthGrant(
                    mcp_server_id=server.id,
                    user_id=user.id,
                    mcp_oauth_client_id=client.id,
                    resource_owner_key=f"xagent:user:{user.id}",
                    issuer="https://auth.example.com",
                    resource="https://mcp.example.com/mcp",
                    scope=f"scope-{index}",
                    access_token=encrypt_value(f"access-{index}"),
                    refresh_token=encrypt_value(f"refresh-{index}"),
                )
            )
        db.add(
            UserOAuth(
                user_id=user.id,
                provider="gmail",
                oauth_provider="google",
                provider_user_id="google-subject",
                access_token="native-access",
                refresh_token="native-refresh",
            )
        )
        db.commit()
        return {"user": user.id, "server": server.id, "client": client.id}


def _records(factory) -> list[CredentialRevocation]:
    with factory() as db:
        rows = db.query(CredentialRevocation).order_by(CredentialRevocation.id).all()
        db.expunge_all()
        return rows


@pytest.mark.parametrize(
    ("table", "key"),
    [("mcp_oauth_clients", "client"), ("mcp_servers", "server")],
)
def test_a_cascade_keeps_the_client_fields(factory, table, key):
    ids = _seed(factory, grants=2)
    with factory() as db:
        db.execute(text(f"DELETE FROM {table} WHERE id = :id"), {"id": ids[key]})
        db.commit()

    records = _records(factory)
    assert len(records) == 2
    for record in records:
        assert record.kind == "mcp_oauth_grant"
        assert record.client_id == "mcp-client"
        assert decrypt_value_strict(record.client_secret) == "mcp-secret"
        assert record.token_endpoint_auth_method == "client_secret_basic"
        assert record.revocation_endpoint == REVOCATION_ENDPOINT
    assert sorted(decrypt_value_strict(r.access_token) for r in records) == [
        "access-0",
        "access-1",
    ]


def test_a_user_cascade_records_native_rows_and_grants(factory):
    ids = _seed(factory)
    with factory() as db:
        db.execute(text("DELETE FROM users WHERE id = :id"), {"id": ids["user"]})
        db.commit()
    assert sorted(r.kind for r in _records(factory)) == [
        "mcp_oauth_grant",
        "user_oauth",
    ]


def test_bulk_revoke_and_bulk_delete_record_each_grant_once(factory):
    _seed(factory, grants=2)
    with factory() as db:
        db.query(MCPOAuthGrant).update(
            {MCPOAuthGrant.status: "revoked"}, synchronize_session=False
        )
        db.commit()
        db.query(MCPOAuthGrant).delete(synchronize_session=False)
        db.query(UserOAuth).delete(synchronize_session=False)
        db.commit()
    assert sorted(r.kind for r in _records(factory)) == [
        "mcp_oauth_grant",
        "mcp_oauth_grant",
        "user_oauth",
    ]


def test_a_delete_racing_a_refresh_records_the_rotated_tokens(factory):
    """The delete waits for the refresh's row lock, then deletes the row the
    refresh committed: the record holds the rotated pair, not the one the
    delete's own snapshot started with."""
    _seed(factory)
    refresh: Session = factory()
    grant = refresh.query(MCPOAuthGrant).with_for_update().one()
    grant.access_token = encrypt_value("rotated-access")
    grant.refresh_token = encrypt_value("rotated-refresh")
    refresh.flush()

    deleted = threading.Event()
    failure: list[BaseException] = []

    def delete() -> None:
        try:
            with factory() as db:
                db.execute(text("DELETE FROM mcp_oauth_grants"))
                db.commit()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)
        finally:
            deleted.set()

    worker = threading.Thread(target=delete)
    worker.start()
    # Blocked on the refresh's row lock.
    assert not deleted.wait(1.0)
    refresh.commit()
    refresh.close()
    worker.join(30)
    assert deleted.is_set() and not failure

    [record] = _records(factory)
    assert decrypt_value_strict(record.access_token) == "rotated-access"
    assert decrypt_value_strict(record.refresh_token) == "rotated-refresh"


def test_a_refresh_after_the_delete_finds_the_grant_gone(factory):
    """The other order: the delete commits first, so the refresh's locked
    re-read finds nothing to write its pair to, and the pair is enqueued
    behind the deletion's record (same grant_key)."""
    from types import SimpleNamespace

    from xagent.web.services.mcp_oauth import _enqueue_tokens_issued_to_a_gone_grant

    ids = _seed(factory)
    with factory() as db:
        grant = db.query(MCPOAuthGrant).one()
        lookup_hash, grant_id = grant.lookup_hash, grant.id
        db.delete(grant)
        db.commit()
    with factory() as db:
        assert (
            db.query(MCPOAuthGrant).filter_by(id=grant_id).with_for_update().first()
            is None
        )
        db.rollback()
        _enqueue_tokens_issued_to_a_gone_grant(
            db,
            grant_id=grant_id,
            lookup_hash=lookup_hash,
            user_id=ids["user"],
            server_id=ids["server"],
            client_id=ids["client"],
            resource_owner_key=f"xagent:user:{ids['user']}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            oauth_client=SimpleNamespace(
                client_id="mcp-client",
                client_secret=encrypt_value("mcp-secret"),
                token_endpoint_auth_method="client_secret_basic",
                metadata_json={"revocation_endpoint": REVOCATION_ENDPOINT},
            ),
            token_data={"access_token": "issued-access", "refresh_token": "issued-rt"},
        )

    deleted, issued = _records(factory)
    assert deleted.grant_key == issued.grant_key
    assert decrypt_value_strict(deleted.access_token) == "access-0"
    assert decrypt_value_strict(issued.access_token) == "issued-access"


def test_replicas_skip_locked_records_and_never_take_two_of_one_grant(factory):
    _seed(factory, grants=2)
    with factory() as db:
        db.query(MCPOAuthGrant).delete(synchronize_session=False)
        db.query(UserOAuth).delete(synchronize_session=False)
        db.commit()
    first, second, native = _records(factory)
    with factory() as db:
        # A second record of the first grant: it must wait for the first.
        db.add(
            CredentialRevocation(
                kind="mcp_oauth_grant",
                source_id=first.source_id,
                user_id=first.user_id,
                grant_key=first.grant_key,
            )
        )
        db.commit()
    now = datetime.now(timezone.utc) + timedelta(seconds=1)

    # Another replica is mid-claim on the first record.
    holder = factory()
    holder.execute(
        text("SELECT id FROM credential_revocations WHERE id = :id FOR UPDATE"),
        {"id": first.id},
    )
    claimed = outbox._claim_due_sync(factory, now=now, limit=10, max_attempts=5)
    holder.rollback()
    holder.close()

    assert sorted(r.id for r in claimed) == sorted([second.id, native.id])
    # Once claimed, the first grant's records stay behind its open record.
    assert [
        r.id for r in outbox._claim_due_sync(factory, now=now, limit=10, max_attempts=5)
    ] == [first.id]
    assert outbox._claim_due_sync(factory, now=now, limit=10, max_attempts=5) == []
