"""Gmail OIDC push verification against real RSA-signed JWTs.

Unlike the fake-verifier tests, these exercise google-auth's actual token
verification (signature, expiry, audience) end-to-end through the unified
callback endpoint: tokens are signed with a locally generated RSA key and
verified against a certs document served over real HTTP. Only the certs URL
differs from production, where Google's own certs endpoint is used.
"""

from __future__ import annotations

import base64
import datetime
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.auth import jwt as google_jwt
from google.auth.crypt import RSASigner
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token

from xagent.web.models.agent import Agent
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerType
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.trigger_providers import (
    GmailProvider,
    register_trigger_provider,
)

from .conftest import _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

KID = "e2e-test-key-1"
MAILBOX = "oidc.real@gmail.example"
CALLBACK_ID = "cb-oidc-real"
AUDIENCE = f"https://stored.example.test/api/triggers/callback/gmail/{CALLBACK_ID}"
PUSH_SERVICE_ACCOUNT = "pubsub-push@e2e-project.iam.gserviceaccount.com"


def _generate_keypair() -> tuple[bytes, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


_SIGNING_PRIVATE, _SIGNING_PUBLIC = _generate_keypair()
_ROGUE_PRIVATE, _ = _generate_keypair()


class _CertsHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({KID: _SIGNING_PUBLIC.decode()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def certs_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CertsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/certs"
    finally:
        server.shutdown()


def _sign_token(
    *,
    private_pem: bytes = _SIGNING_PRIVATE,
    audience: str = AUDIENCE,
    issuer: str = "https://accounts.google.com",
    email: str = PUSH_SERVICE_ACCOUNT,
    email_verified: bool = True,
    expired: bool = False,
) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    issued = now - (datetime.timedelta(hours=2) if expired else datetime.timedelta())
    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": "1234567890",
        "email": email,
        "email_verified": email_verified,
        "iat": int(issued.timestamp()),
        "exp": int((issued + datetime.timedelta(hours=1)).timestamp()),
    }
    signer = RSASigner.from_string(private_pem.decode(), KID)
    return google_jwt.encode(signer, payload).decode()


def _real_verifier(certs_url: str):
    """google-auth verification with only the certs URL swapped for local."""

    def verify(token: str, audience: str):
        return id_token.verify_token(
            token,
            GoogleAuthRequest(),
            audience=audience,
            certs_url=certs_url,
        )

    return verify


class _EmptyHistoryGmailService:
    class _Exec:
        def __init__(self, payload):
            self._payload = payload

        def execute(self):
            return self._payload

    def users(self):
        return self

    def history(self):
        return self

    def list(self, **_kwargs):
        return self._Exec({"history": []})


def _seed_gmail_state() -> int:
    db = _direct_db_session()
    try:
        user = User(username="oidc-real-user", password_hash="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        agent = Agent(
            user_id=int(user.id),
            name="OIDC agent",
            description="d",
            instructions="i",
            execution_mode="balanced",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        oauth = UserOAuth(
            user_id=int(user.id),
            provider="gmail",
            access_token="tok",
            email=MAILBOX,
        )
        db.add(oauth)
        db.commit()
        db.refresh(oauth)
        trigger = AgentTrigger(
            user_id=int(user.id),
            agent_id=int(agent.id),
            type=TriggerType.GMAIL.value,
            provider=TriggerType.GMAIL.value,
            name="OIDC trigger",
            enabled=True,
            resource_id=MAILBOX,
            config={"watch_label": "INBOX", "oauth_account_id": int(oauth.id)},
        )
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email=MAILBOX,
            history_id="100",
            topic_name="projects/e2e/topics/t",
            callback_id=CALLBACK_ID,
            push_audience=AUDIENCE,
        )
        db.add_all([trigger, state])
        db.commit()
        return int(trigger.id)
    finally:
        db.close()


@pytest.fixture()
def real_crypto_provider(certs_url, monkeypatch):
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT", PUSH_SERVICE_ACCOUNT)
    provider = GmailProvider(
        service_factory=lambda _db, _oauth: _EmptyHistoryGmailService(),
        oidc_verifier=_real_verifier(certs_url),
    )
    register_trigger_provider(provider, replace=True)
    try:
        yield provider
    finally:
        register_trigger_provider(GmailProvider(), replace=True)


def _push_body() -> dict:
    data = base64.urlsafe_b64encode(
        json.dumps({"emailAddress": MAILBOX, "historyId": "222"}).encode()
    ).decode()
    return {"message": {"data": data, "messageId": "m-oidc-real"}}


def _post(token: str) -> object:
    return client.post(
        f"/api/triggers/callback/gmail/{CALLBACK_ID}",
        headers={"Authorization": f"Bearer {token}"},
        json=_push_body(),
    )


def test_valid_google_style_jwt_is_accepted(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token())
    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "accepted"


def test_token_signed_by_unknown_key_is_rejected(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token(private_pem=_ROGUE_PRIVATE))
    # Gmail acks rejections with 200 to stop Pub/Sub redelivery, but the
    # outcome records the signature rejection and no run is created.
    assert response.status_code == 200
    assert response.json()["outcome"] == "rejected_signature"
    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
    finally:
        db.close()


def test_wrong_audience_is_rejected(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token(audience="https://other.example.test/api/callback"))
    assert response.status_code == 200
    assert response.json()["outcome"] == "rejected_signature"


def test_expired_token_is_rejected(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token(expired=True))
    assert response.status_code == 200
    assert response.json()["outcome"] == "rejected_signature"


def test_untrusted_issuer_is_rejected(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token(issuer="https://evil.example.test"))
    assert response.status_code == 200
    assert response.json()["outcome"] == "rejected_signature"


def test_service_account_email_mismatch_is_rejected(real_crypto_provider) -> None:
    _seed_gmail_state()
    response = _post(_sign_token(email="someone-else@example.iam.gserviceaccount.com"))
    assert response.status_code == 200
    assert response.json()["outcome"] == "rejected_signature"
