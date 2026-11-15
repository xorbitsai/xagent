"""The credential revocation outbox on SQLite: the triggers and the worker.

The triggers are exercised through every way a row can go -- ORM delete, bulk
delete, raw SQL, a foreign-key cascade from the user, the MCP server or the
OAuth client, and an MCP grant's status flip to ``revoked`` -- and the worker
through each outcome, with only provider HTTP faked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import decrypt_value_strict, encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.credential_revocation import (
    CredentialRevocation,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import COMPOSIO_TOKEN_TYPE, UserOAuth
from xagent.web.services import credential_revocations as outbox
from xagent.web.services import mcp_oauth as mcp_oauth_service

REVOCATION_ENDPOINT = "https://auth.example.com/revoke"


@pytest.fixture()
def factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'outbox.db'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    engine.dispose()


def _user(db: Session, name: str = "alice") -> User:
    user = User(username=name, password_hash="x", is_admin=False)
    db.add(user)
    db.flush()
    return user


def _native(
    db: Session,
    user: User,
    *,
    provider: str = "gmail",
    oauth_provider: str | None = "google",
    subject: str | None = "google-subject",
    access_token: str = "native-access",
    refresh_token: str | None = "native-refresh",
    token_type: str | None = "Bearer",
) -> UserOAuth:
    row = UserOAuth(
        user_id=user.id,
        provider=provider,
        oauth_provider=oauth_provider,
        provider_user_id=subject,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type=token_type,
    )
    db.add(row)
    db.flush()
    return row


def _mcp(db: Session, user: User, *, status: str = "active", **grant: Any):
    server = MCPServer.from_config(
        {
            "name": f"records-{user.username}-{grant.get('scope', 'read')}",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "auth": {"type": "mcp_oauth"},
        }
    )
    db.add(server)
    db.flush()
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="mcp-client",
        client_secret=encrypt_value("mcp-secret"),
        token_endpoint_auth_method="client_secret_post",
        redirect_uri="https://app.example.com/api/mcp/oauth/callback",
        metadata_json={"revocation_endpoint": REVOCATION_ENDPOINT},
    )
    db.add(client)
    db.flush()
    row = _grant(db, user, server, client, status=status, **grant)
    return server, client, row


def _grant(db, user, server, client, *, status="active", scope="read", **extra):
    row = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope=scope,
        access_token=encrypt_value(extra.get("access_token", "mcp-access")),
        refresh_token=encrypt_value(extra.get("refresh_token", "mcp-refresh")),
        status=status,
    )
    db.add(row)
    db.flush()
    return row


def _records(factory) -> list[CredentialRevocation]:
    with factory() as db:
        rows = db.query(CredentialRevocation).order_by(CredentialRevocation.id).all()
        db.expunge_all()
        return rows


# --- the triggers -------------------------------------------------------------


def test_orm_delete_of_a_native_row_records_its_stored_tokens(factory):
    with factory() as db:
        row = _native(db, _user(db))
        stored_access = db.execute(
            text("SELECT access_token FROM user_oauth WHERE id = :id"), {"id": row.id}
        ).scalar_one()
        db.delete(row)
        db.commit()

    [record] = _records(factory)
    assert record.kind == "user_oauth"
    assert record.status == "pending"
    assert record.issuer == "google"
    assert record.provider_user_id == "google-subject"
    assert record.grant_key == "user_oauth:google:google-subject"
    # Copied as stored -- the ciphertext itself -- and opened by the worker.
    assert record.access_token == stored_access
    assert decrypt_value_strict(record.access_token) == "native-access"
    assert decrypt_value_strict(record.refresh_token) == "native-refresh"
    assert record.attempts == 0
    assert record.next_attempt_at is not None


def test_bulk_and_raw_sql_deletes_record_too(factory):
    with factory() as db:
        user = _user(db)
        _native(db, user, provider="gmail")
        _native(db, user, provider="google-drive", subject="other")
        third = _native(db, user, provider="github", oauth_provider="github")
        db.commit()
        db.query(UserOAuth).filter(UserOAuth.provider != "github").delete(
            synchronize_session=False
        )
        db.execute(text("DELETE FROM user_oauth WHERE id = :id"), {"id": third.id})
        db.commit()

    assert sorted(r.issuer for r in _records(factory)) == ["github", "google", "google"]


def test_composio_rows_are_not_recorded(factory):
    with factory() as db:
        row = _native(
            db, _user(db), token_type=COMPOSIO_TOKEN_TYPE, oauth_provider=None
        )
        db.delete(row)
        db.commit()
    assert _records(factory) == []


def test_unknown_issuer_is_still_recorded(factory):
    with factory() as db:
        row = _native(db, _user(db), oauth_provider=None)
        row_id = row.id
        db.delete(row)
        db.commit()
    [record] = _records(factory)
    assert record.issuer is None
    assert record.grant_key == f"user_oauth#{row_id}"


def test_user_cascade_records_native_rows_and_grants(factory):
    with factory() as db:
        user = _user(db)
        _native(db, user)
        _mcp(db, user)
        db.commit()
        db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
        db.commit()
    assert sorted(r.kind for r in _records(factory)) == [
        "mcp_oauth_grant",
        "user_oauth",
    ]


def test_mcp_grant_delete_records_tokens_client_and_identity(factory):
    with factory() as db:
        user = _user(db)
        server, client, grant = _mcp(db, user)
        ids = (grant.id, grant.lookup_hash, server.id, client.id, user.id)
        db.delete(grant)
        db.commit()
    grant_id, lookup_hash, server_id, client_id, user_id = ids

    [record] = _records(factory)
    assert record.kind == "mcp_oauth_grant"
    assert record.source_id == grant_id
    assert record.grant_key == f"mcp_oauth_grant:{lookup_hash}"
    assert decrypt_value_strict(record.access_token) == "mcp-access"
    assert decrypt_value_strict(record.refresh_token) == "mcp-refresh"
    assert record.client_id == "mcp-client"
    assert decrypt_value_strict(record.client_secret) == "mcp-secret"
    assert record.token_endpoint_auth_method == "client_secret_post"
    assert record.revocation_endpoint == REVOCATION_ENDPOINT
    assert (record.mcp_server_id, record.mcp_oauth_client_id) == (server_id, client_id)
    assert record.resource_owner_key == f"xagent:user:{user_id}"
    assert record.issuer == "https://auth.example.com"
    assert record.resource == "https://mcp.example.com/mcp"


@pytest.mark.parametrize("parent", ["mcp_oauth_clients", "mcp_servers"])
def test_cascade_from_the_client_or_server_keeps_the_client_fields(factory, parent):
    """The BEFORE DELETE trigger on the client deletes its grants while the
    client row still exists, so the grant's record still finds it."""
    with factory() as db:
        server, client, _ = _mcp(db, _user(db))
        db.commit()
        parent_id = client.id if parent == "mcp_oauth_clients" else server.id
        db.execute(text(f"DELETE FROM {parent} WHERE id = :id"), {"id": parent_id})
        db.commit()
        assert db.query(MCPOAuthGrant).count() == 0

    [record] = _records(factory)
    assert record.client_id == "mcp-client"
    assert record.revocation_endpoint == REVOCATION_ENDPOINT
    assert decrypt_value_strict(record.client_secret) == "mcp-secret"


def test_status_flip_to_revoked_records_once(factory):
    with factory() as db:
        _, _, grant = _mcp(db, _user(db))
        db.commit()
        grant.status = "revoked"
        db.commit()
        # Re-saving a revoked grant, and finally deleting it, adds nothing:
        # its tokens cannot change while it is revoked.
        grant.revoked_at = datetime.now(timezone.utc)
        db.commit()
        db.delete(grant)
        db.commit()
    [record] = _records(factory)
    assert decrypt_value_strict(record.access_token) == "mcp-access"


def test_bulk_status_update_records_each_grant(factory):
    with factory() as db:
        user = _user(db)
        server, client, _ = _mcp(db, user)
        _grant(db, user, server, client, scope="write")
        db.commit()
        db.query(MCPOAuthGrant).update(
            {MCPOAuthGrant.status: "revoked"}, synchronize_session=False
        )
        db.commit()
    assert len(_records(factory)) == 2


def test_a_rolled_back_delete_records_nothing(factory):
    with factory() as db:
        row = _native(db, _user(db))
        db.commit()
        db.delete(row)
        db.flush()
        db.rollback()
    assert _records(factory) == []


def test_the_record_holds_the_tokens_of_the_last_committed_refresh(factory):
    """A refresh that rotated the tokens before the delete is what the record
    holds -- there is no earlier snapshot for it to be stale against."""
    with factory() as db:
        _, _, grant = _mcp(db, _user(db))
        row = _native(db, db.query(User).one())
        db.commit()
        grant.access_token = encrypt_value("rotated-access")
        grant.refresh_token = encrypt_value("rotated-refresh")
        row.access_token = "rotated-native-access"
        db.commit()
        db.delete(grant)
        db.delete(row)
        db.commit()
    tokens = {r.kind: decrypt_value_strict(r.access_token) for r in _records(factory)}
    assert tokens == {
        "mcp_oauth_grant": "rotated-access",
        "user_oauth": "rotated-native-access",
    }


# --- tokens dropped in place --------------------------------------------------


def _drop(factory, **updates) -> list[CredentialRevocation]:
    """Update one stored Google row in place and return the records."""
    with factory() as db:
        row = _native(db, _user(db))
        db.commit()
        for field, value in updates.items():
            setattr(row, field, value)
        db.commit()
    return _records(factory)


def test_a_reconnect_tombstone_records_the_dropped_tokens(factory):
    """What a permanently failed refresh leaves (tools.config, Gmail
    provisioning): the row stays, its tokens go."""
    [record] = _drop(factory, access_token="", refresh_token=None)

    assert record.kind == "user_oauth"
    assert decrypt_value_strict(record.access_token) == "native-access"
    assert decrypt_value_strict(record.refresh_token) == "native-refresh"
    assert record.grant_key == "user_oauth:google:google-subject"


def test_dropping_only_the_refresh_token_records_it(factory):
    [record] = _drop(factory, refresh_token=None)
    assert decrypt_value_strict(record.refresh_token) == "native-refresh"


def test_dropping_the_access_token_of_a_row_without_a_refresh_token_records_it(
    factory,
):
    [record] = _drop(factory, refresh_token=None)
    with factory() as db:
        row = db.query(UserOAuth).one()
        row.access_token = ""
        db.commit()
    assert len(_records(factory)) == 2
    assert decrypt_value_strict(_records(factory)[1].access_token) == "native-access"


@pytest.mark.parametrize(
    "updates",
    [
        # A refresh that rotates both tokens.
        {"access_token": "rotated-access", "refresh_token": "rotated-refresh"},
        # An access token cleared while the refresh token stays: the row can
        # still mint access tokens, so nothing was dropped.
        {"access_token": ""},
        # A refresh that only replaces the access token.
        {"access_token": "new-access"},
    ],
    ids=["rotation", "access-cleared-refresh-kept", "access-rotated"],
)
def test_rotations_and_partial_clears_record_nothing(factory, updates):
    assert _drop(factory, **updates) == []


def test_a_callback_that_blanks_then_refills_in_one_flush_records_nothing(factory):
    """The OAuth callback sets ``refresh_token`` to None and then to the new
    value before anything is flushed: one UPDATE with the final value."""
    with factory() as db:
        row = _native(db, _user(db))
        db.commit()
        row.refresh_token = None
        row.refresh_token = "reconnected-refresh"
        row.access_token = "reconnected-access"
        db.commit()
    assert _records(factory) == []


def test_a_composio_tombstone_records_nothing(factory):
    with factory() as db:
        row = _native(
            db,
            _user(db),
            token_type=COMPOSIO_TOKEN_TYPE,
            oauth_provider=None,
            refresh_token=None,
        )
        db.commit()
        row.access_token = ""
        db.commit()
    assert _records(factory) == []


def test_a_bulk_clear_records_each_row(factory):
    with factory() as db:
        user = _user(db)
        _native(db, user, provider="gmail")
        _native(db, user, provider="google-drive", subject="other")
        db.commit()
        db.query(UserOAuth).update(
            {UserOAuth.access_token: "", UserOAuth.refresh_token: None},
            synchronize_session=False,
        )
        db.commit()
    assert len(_records(factory)) == 2


# --- the worker ---------------------------------------------------------------


@pytest.fixture()
def now() -> datetime:
    """The worker's clock starts from the real one, as the revocation fence
    reads that; tests move it forward to cross backoffs and leases. One
    second ahead, because SQLite's CURRENT_TIMESTAMP (a new record's due
    time) has whole seconds."""
    return datetime.now(timezone.utc) + timedelta(seconds=1)


@pytest.fixture(autouse=True)
def _no_fence_close_margin(monkeypatch):
    """A closed fence keeps refusing for a few real seconds of clock-skew
    margin; the tests retry sooner than that in real time."""
    from xagent.web.services import oauth_revocation_fences

    monkeypatch.setattr(
        oauth_revocation_fences, "OAUTH_REVOCATION_CLOSE_MARGIN", timedelta(0)
    )


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


@pytest.fixture()
def google(monkeypatch):
    """Google's revoke endpoint; ``answers`` queues responses, default 200."""
    calls: list[dict[str, str]] = []
    answers: list[_Response] = []

    def post(url, data=None, timeout=None, **_):
        assert url == "https://oauth2.googleapis.com/revoke"
        calls.append(dict(data))
        return answers.pop(0) if answers else _Response(200)

    monkeypatch.setattr(auth_api.requests, "post", post)
    return SimpleNamespace(calls=calls, answers=answers)


@pytest.fixture()
def rfc7009(monkeypatch):
    """An RFC 7009 endpoint behind the MCP OAuth HTTP client."""
    calls: list[dict[str, str]] = []
    answers: list[httpx.Response] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == REVOCATION_ENDPOINT
        calls.append(dict(httpx.QueryParams(request.content.decode())))
        return answers.pop(0) if answers else httpx.Response(200)

    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **_: real_client(transport=httpx.MockTransport(handler)),
    )
    return SimpleNamespace(calls=calls, answers=answers)


async def _run(factory, *, at: datetime | None = None, **kwargs):
    at = at or datetime.now(timezone.utc) + timedelta(seconds=1)
    return await outbox.run_credential_revocation_batch(
        factory, now=lambda: at, **kwargs
    )


def _delete_native(factory, **row) -> None:
    with factory() as db:
        user = db.query(User).filter_by(username=row.pop("user", "alice")).first()
        account = _native(db, user or _user(db, "alice"), **row)
        db.delete(account)
        db.commit()


def _assert_erased(record: CredentialRevocation) -> None:
    assert (record.access_token, record.refresh_token, record.client_secret) == (
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_worker_revokes_a_google_grant_through_its_refresh_token(factory, google):
    _delete_native(factory)

    report = await _run(factory)

    assert report.outcomes == {"revoked": 1}
    assert google.calls == [{"token": "native-refresh"}]
    [record] = _records(factory)
    assert record.status == "revoked"
    assert record.attempts == 1
    _assert_erased(record)


@pytest.mark.asyncio
async def test_already_revoked_counts_as_done(factory, google, monkeypatch):
    google.answers.append(_Response(400, '{"error": "invalid_token"}'))
    _delete_native(factory)
    with factory() as db:
        db.add(
            OAuthProvider(
                provider_name="github",
                name="GitHub",
                client_id=encrypt_value("gh-client"),
                client_secret=encrypt_value("gh-secret"),
                auth_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
            )
        )
        db.commit()
    _delete_native(factory, provider="github", oauth_provider="github", subject="42")
    monkeypatch.setattr(auth_api.requests, "delete", lambda *a, **k: _Response(404))

    report = await _run(factory)

    assert report.outcomes == {"revoked": 2}
    assert all(r.status == "revoked" and r.last_error for r in _records(factory))


@pytest.mark.asyncio
async def test_a_provider_failure_backs_off_then_dies(factory, now, google, caplog):
    google.answers.extend([_Response(503), _Response(503), _Response(503)])
    _delete_native(factory)

    await _run(factory, at=now, max_attempts=3)
    [record] = _records(factory)
    assert (record.status, record.attempts) == ("pending", 1)
    assert "HTTP 503" in record.last_error
    assert outbox._aware(record.next_attempt_at) == now + timedelta(minutes=1)
    assert record.access_token is not None

    # Not due before its backoff ends.
    assert (await _run(factory, at=now + timedelta(seconds=30))).claimed == 0

    second = now + timedelta(minutes=1)
    await _run(factory, at=second, max_attempts=3)
    [record] = _records(factory)
    assert record.attempts == 2
    assert outbox._aware(record.next_attempt_at) == second + timedelta(minutes=2)

    with caplog.at_level("ERROR"):
        await _run(factory, at=second + timedelta(minutes=2), max_attempts=3)
    [record] = _records(factory)
    assert (record.status, record.attempts) == ("dead", 3)
    _assert_erased(record)
    assert any("is dead after 3 attempts" in m for m in caplog.messages)
    assert len(google.calls) == 3


@pytest.mark.asyncio
async def test_unknown_issuer_fails_without_guessing(factory, google):
    _delete_native(factory, oauth_provider=None)

    assert (await _run(factory)).outcomes == {"failed": 1}

    [record] = _records(factory)
    assert record.status == "failed"
    assert "unknown issuer" in record.last_error
    _assert_erased(record)
    assert google.calls == []


@pytest.mark.asyncio
async def test_providers_without_revocation_are_skipped(factory, google):
    _delete_native(factory, provider="notion", oauth_provider="notion")

    assert (await _run(factory)).outcomes == {"skipped": 1}
    assert google.calls == []


@pytest.mark.asyncio
async def test_skipped_while_referenced_then_revoked_with_the_last_reference(
    factory, google
):
    """Google revokes the whole grant: while Bob's Drive row still holds the
    same Google account, Alice's deletion must not revoke it. Bob's own
    deletion enqueues later and revokes it then."""
    with factory() as db:
        alice, bob = _user(db, "alice"), _user(db, "bob")
        alice_row = _native(db, alice, access_token="a", refresh_token="alice-rt")
        bob_row_id = _native(
            db, bob, provider="google-drive", access_token="b", refresh_token="bob-rt"
        ).id
        db.commit()
        db.delete(alice_row)
        db.commit()

    assert (await _run(factory)).outcomes == {"skipped": 1}
    assert google.calls == []

    with factory() as db:
        db.delete(db.get(UserOAuth, bob_row_id))
        db.commit()
    assert (await _run(factory)).outcomes == {"revoked": 1}
    assert google.calls == [{"token": "bob-rt"}]
    assert [r.status for r in _records(factory)] == ["skipped", "revoked"]
    for record in _records(factory):
        _assert_erased(record)


@pytest.mark.asyncio
async def test_an_open_fence_defers_without_spending_an_attempt(factory, google):
    from xagent.web.services.oauth_revocation_fences import (
        lock_oauth_revocation_fence,
        open_oauth_revocation_fence,
    )

    _delete_native(factory)
    with factory() as db:
        deadline = open_oauth_revocation_fence(
            lock_oauth_revocation_fence(
                db, provider="google", provider_user_id="google-subject"
            )
        )
        db.commit()

    report = await _run(factory)

    assert report.outcomes == {"pending": 1}
    [record] = _records(factory)
    assert (record.status, record.attempts) == ("pending", 0)
    assert outbox._aware(record.next_attempt_at) == deadline
    assert google.calls == []


@pytest.mark.asyncio
async def test_records_of_one_grant_run_one_at_a_time_oldest_first(
    factory, now, google
):
    google.answers.append(_Response(503))
    with factory() as db:
        user = _user(db)
        first = _native(db, user, access_token="a1", refresh_token="first")
        second = _native(db, user, provider="google-drive", refresh_token="second")
        db.commit()
        db.delete(first)
        db.delete(second)
        db.commit()

    # The first fails and backs off; the second waits behind it.
    assert (await _run(factory, at=now)).claimed == 1
    assert google.calls == [{"token": "first"}]
    assert (await _run(factory, at=now + timedelta(seconds=59))).claimed == 0

    later = now + timedelta(minutes=1)
    assert (await _run(factory, at=later)).outcomes == {"revoked": 1}
    assert (await _run(factory, at=later)).outcomes == {"revoked": 1}
    assert google.calls[1:] == [{"token": "first"}, {"token": "second"}]


@pytest.mark.asyncio
async def test_a_claim_is_leased_and_a_lapsed_lease_is_retried_then_dies(factory, now):
    _delete_native(factory)

    claimed = outbox._claim_due_sync(factory, now=now, limit=10, max_attempts=2)
    assert [r.attempts for r in claimed] == [1]
    # Leased: no one else claims it until the lease lapses.
    assert outbox._claim_due_sync(factory, now=now, limit=10, max_attempts=2) == []

    lapsed = now + outbox.CLAIM_LEASE
    assert [
        r.attempts
        for r in outbox._claim_due_sync(factory, now=lapsed, limit=10, max_attempts=2)
    ] == [2]
    # A third lapse with the budget spent: dead, without another attempt.
    assert (
        outbox._claim_due_sync(
            factory, now=lapsed + outbox.CLAIM_LEASE, limit=10, max_attempts=2
        )
        == []
    )
    [record] = _records(factory)
    assert record.status == "dead"
    _assert_erased(record)


@pytest.mark.asyncio
async def test_an_mcp_grant_is_revoked_per_rfc7009(factory, rfc7009):
    with factory() as db:
        _, _, grant = _mcp(db, _user(db))
        db.delete(grant)
        db.commit()

    assert (await _run(factory)).outcomes == {"revoked": 1}

    assert rfc7009.calls == [
        {
            "client_id": "mcp-client",
            "client_secret": "mcp-secret",
            "token": "mcp-access",
            "token_type_hint": "access_token",
        },
        {
            "client_id": "mcp-client",
            "client_secret": "mcp-secret",
            "token": "mcp-refresh",
            "token_type_hint": "refresh_token",
        },
    ]
    [record] = _records(factory)
    _assert_erased(record)


@pytest.mark.asyncio
async def test_an_mcp_invalid_token_answer_is_done_and_a_5xx_retries(
    factory, now, rfc7009
):
    rfc7009.answers.extend(
        [
            httpx.Response(400, json={"error": "invalid_token"}),
            httpx.Response(500),
        ]
    )
    with factory() as db:
        _, _, grant = _mcp(db, _user(db))
        db.delete(grant)
        db.commit()

    assert (await _run(factory, at=now)).outcomes == {"pending": 1}
    [record] = _records(factory)
    assert "HTTP 500 for the refresh_token" in record.last_error

    assert (await _run(factory, at=now + timedelta(minutes=1))).outcomes == {
        "revoked": 1
    }


@pytest.mark.asyncio
async def test_an_mcp_grant_with_an_active_successor_is_skipped(factory, rfc7009):
    """A superseded grant (same server, owner, client, issuer and resource)
    must not be revoked: the provider may end the successor's authorization
    with it."""
    with factory() as db:
        user = _user(db)
        server, client, old = _mcp(db, user)
        _grant(db, user, server, client, scope="read write")
        db.delete(old)
        db.commit()

    assert (await _run(factory)).outcomes == {"skipped": 1}
    assert rfc7009.calls == []


@pytest.mark.asyncio
async def test_an_mcp_server_without_a_revocation_endpoint_is_skipped(factory, rfc7009):
    with factory() as db:
        _, client, grant = _mcp(db, _user(db))
        client.metadata_json = {}
        db.flush()
        db.delete(grant)
        db.commit()

    assert (await _run(factory)).outcomes == {"skipped": 1}


@pytest.mark.asyncio
async def test_refresh_compensation_enqueues_the_issued_pair(factory, rfc7009):
    with factory() as db:
        user = _user(db)
        server, client, grant = _mcp(db, user)
        ids = SimpleNamespace(
            grant=grant.id,
            hash=grant.lookup_hash,
            user=user.id,
            server=server.id,
            client=client.id,
        )
        snapshot = SimpleNamespace(
            client_id=client.client_id,
            client_secret=client.client_secret,
            token_endpoint_auth_method=client.token_endpoint_auth_method,
            metadata_json=client.metadata_json,
        )
        db.delete(grant)
        db.commit()

    with factory() as db:
        mcp_oauth_service._enqueue_tokens_issued_to_a_gone_grant(
            db,
            grant_id=ids.grant,
            lookup_hash=ids.hash,
            user_id=ids.user,
            server_id=ids.server,
            client_id=ids.client,
            resource_owner_key=f"xagent:user:{ids.user}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            oauth_client=snapshot,
            token_data={"access_token": "new-access", "refresh_token": "new-rt"},
        )

    deleted, issued = _records(factory)
    assert issued.grant_key == deleted.grant_key
    assert decrypt_value_strict(issued.access_token) == "new-access"
    assert decrypt_value_strict(issued.refresh_token) == "new-rt"

    # One grant at a time: the deletion's record first, then the issued pair.
    await _run(factory)
    await _run(factory)
    assert [call["token"] for call in rfc7009.calls] == [
        "mcp-access",
        "mcp-refresh",
        "new-access",
        "new-rt",
    ]


@pytest.mark.asyncio
async def test_a_tombstoned_grant_is_revoked_once_nothing_references_it(
    factory, google
):
    """The reconnect tombstone left nothing usable on the row, so the worker
    revokes the dropped refresh token (Google answers ``invalid_token`` if
    the grant is already dead, which also settles the record)."""
    _drop(factory, access_token="", refresh_token=None)

    assert (await _run(factory)).outcomes == {"revoked": 1}
    assert google.calls == [{"token": "native-refresh"}]


@pytest.mark.asyncio
async def test_a_drop_from_a_row_that_kept_a_live_token_is_skipped(factory, google):
    """A row whose refresh token went but whose access token stayed (a
    reconnect without a new refresh token, a mirror re-copied from a shared
    credential without one) still holds the account's grant: revoking the
    dropped token would kill it."""
    _drop(factory, refresh_token=None)

    assert (await _run(factory)).outcomes == {"skipped": 1}
    assert google.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1/revoke", "ftp://auth.example.com/revoke"],
    ids=["local-address", "not-http"],
)
async def test_an_endpoint_the_url_policy_refuses_fails_at_once(
    factory, rfc7009, endpoint
):
    with factory() as db:
        _, client, grant = _mcp(db, _user(db))
        client.metadata_json = {"revocation_endpoint": endpoint}
        db.flush()
        db.delete(grant)
        db.commit()

    assert (await _run(factory)).outcomes == {"failed": 1}

    [record] = _records(factory)
    assert record.attempts == 1
    assert "refused by the OAuth URL policy" in record.last_error
    assert endpoint not in record.last_error
    assert rfc7009.calls == []
    _assert_erased(record)


@pytest.mark.asyncio
async def test_a_live_row_with_an_unknown_issuer_blocks_the_revoke(factory, google):
    """A custom-app row the issuer backfill could not prove stays NULL. Nothing
    proves it was not issued by Google for the same account, so while it holds
    a token the Google record is skipped rather than risk cutting it off."""
    with factory() as db:
        alice, bob = _user(db, "alice"), _user(db, "bob")
        _native(db, bob, provider="acme-mail", oauth_provider=None)
        doomed = _native(db, alice)
        db.commit()
        db.delete(doomed)
        db.commit()

    assert (await _run(factory)).outcomes == {"skipped": 1}
    assert google.calls == []
    [record] = _records(factory)
    assert "another local connection" in record.last_error


@pytest.mark.asyncio
async def test_a_tombstoned_row_with_an_unknown_issuer_fails_without_a_call(
    factory, google
):
    """Its dropped tokens are recorded like any other drop, and the worker
    refuses to guess where to send them."""
    with factory() as db:
        row = _native(db, _user(db), oauth_provider=None)
        db.commit()
        row.access_token, row.refresh_token = "", None
        db.commit()

    assert (await _run(factory)).outcomes == {"failed": 1}
    [record] = _records(factory)
    assert record.issuer is None
    assert "unknown issuer" in record.last_error
    _assert_erased(record)
    assert google.calls == []
