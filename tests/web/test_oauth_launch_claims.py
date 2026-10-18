"""Durable launch-ticket consumption does not depend on a process-local cache."""

import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web.models.database import Base
from xagent.web.models.oidc_consumed_token import OidcConsumedToken
from xagent.web.services.oauth_launch import consume_oauth_launch_ticket


def _claims(**overrides):
    return {
        "jti": secrets.token_urlsafe(24),
        "exp": (datetime.now(timezone.utc) + timedelta(seconds=30)).timestamp(),
        **overrides,
    }


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'claims.db'}", connect_args={"timeout": 30}
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def test_concurrent_independent_sessions_consume_exactly_once(engine):
    proof = _claims()

    def claim(_):
        with Session(engine) as db:
            return consume_oauth_launch_ticket(db, proof)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(16))) == 1
    with Session(engine) as db:
        assert db.query(OidcConsumedToken).count() == 1
        assert not consume_oauth_launch_ticket(db, proof)


def test_reopening_engine_does_not_release_consumed_proof(engine):
    proof = _claims()
    with Session(engine) as db:
        assert consume_oauth_launch_ticket(db, proof)
        db.rollback()
    engine.dispose()
    with Session(engine) as db:
        assert not consume_oauth_launch_ticket(db, proof)


@pytest.mark.parametrize(
    "expires", [None, True, "123", {}, float("inf"), float("nan"), 0, -1, 10**100]
)
def test_invalid_or_expired_claims_cannot_consume(engine, expires):
    with Session(engine) as db:
        assert not consume_oauth_launch_ticket(db, _claims(exp=expires))
        assert db.query(OidcConsumedToken).count() == 0


@pytest.mark.parametrize(
    "token_id", [None, True, "", "short", "a" * 96, "a" * 31 + ":"]
)
def test_malformed_ids_cannot_allocate_rows(engine, token_id):
    with Session(engine) as db:
        assert not consume_oauth_launch_ticket(db, _claims(jti=token_id))
        assert db.query(OidcConsumedToken).count() == 0


def test_launch_cleanup_does_not_remove_other_proof_namespaces(engine):
    with Session(engine) as db:
        old = datetime.now(timezone.utc) - timedelta(days=1)
        db.add_all(
            [
                OidcConsumedToken(token_id="oauth-state:other", expires_at=old),
                OidcConsumedToken(token_id="oidc-exchange-other", expires_at=old),
                OidcConsumedToken(token_id="oauth-launch:old", expires_at=old),
            ]
        )
        db.commit()
        assert consume_oauth_launch_ticket(db, _claims())
        ids = {row.token_id for row in db.query(OidcConsumedToken).all()}
        assert "oauth-state:other" in ids
        assert "oidc-exchange-other" in ids
        assert "oauth-launch:old" not in ids
