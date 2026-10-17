import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from xagent.web.api import auth, oidc_google
from xagent.web.models.database import Base
from xagent.web.models.oidc_consumed_token import OidcConsumedToken
from xagent.web.services.oauth_state_claims import claim_oauth_state


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'states.db'}", connect_args={"timeout": 15}
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def test_claim_is_durable_across_sessions_and_engine_restart(sessions):
    expiry = time.time() + 600
    with sessions() as db:
        assert claim_oauth_state(db, "signed-state", expires_at=expiry)
        assert db.query(OidcConsumedToken).one().token_id.startswith("oauth-state:")
    url = sessions.kw["bind"].url
    sessions.kw["bind"].dispose()
    restarted = create_engine(url)
    try:
        with sessionmaker(bind=restarted)() as db:
            assert not claim_oauth_state(db, "signed-state", expires_at=expiry)
    finally:
        restarted.dispose()


def test_concurrent_workers_can_only_claim_once(sessions):
    expiry = time.time() + 600

    def claim(_):
        with sessions() as db:
            return claim_oauth_state(db, "same-state", expires_at=expiry)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(16))) == 1


def test_later_provider_rollback_does_not_release_the_claim(sessions):
    expiry = time.time() + 600
    with sessions() as db:
        assert claim_oauth_state(db, "failed-provider-state", expires_at=expiry)
        db.rollback()
    with sessions() as db:
        assert not claim_oauth_state(db, "failed-provider-state", expires_at=expiry)


@pytest.mark.parametrize(
    "expiry",
    [
        None,
        True,
        "9999999999",
        float("nan"),
        float("inf"),
        -1,
        pytest.param(10**1000, id="overflow"),
    ],
)
def test_invalid_or_expired_states_do_not_claim(sessions, expiry):
    with sessions() as db:
        assert not claim_oauth_state(db, "invalid", expires_at=expiry)
        assert db.query(OidcConsumedToken).count() == 0


def test_outage_fails_closed_without_process_local_fallback():
    db = Mock()
    db.execute.side_effect = OperationalError("statement", {}, Exception("unavailable"))
    with pytest.raises(OperationalError):
        claim_oauth_state(db, "state", expires_at=time.time() + 600)
    db.rollback.assert_called_once()


@pytest.mark.parametrize("operation", ["add", "commit"])
def test_write_failure_rolls_back_and_propagates(sessions, monkeypatch, operation):
    with sessions() as db:

        def fail(*args, **kwargs):
            raise OperationalError("statement", {}, Exception("unavailable"))

        monkeypatch.setattr(db, operation, fail)
        with pytest.raises(OperationalError):
            claim_oauth_state(db, "signed", expires_at=time.time() + 600)
    with sessions() as db:
        assert db.query(OidcConsumedToken).count() == 0


def test_cleanup_preserves_other_namespace_even_with_fast_oidc_clock(
    sessions, monkeypatch
):
    now = datetime.now(timezone.utc)
    with sessions() as db:
        db.add_all(
            [
                OidcConsumedToken(
                    token_id="oauth-state:expired", expires_at=now - timedelta(hours=1)
                ),
                OidcConsumedToken(
                    token_id="oidc-expired", expires_at=now - timedelta(hours=1)
                ),
                OidcConsumedToken(
                    token_id="oauth-state:live", expires_at=now + timedelta(minutes=5)
                ),
            ]
        )
        db.commit()
        assert claim_oauth_state(db, "fresh", expires_at=time.time() + 600)
        assert db.get(OidcConsumedToken, "oauth-state:expired") is None
        assert db.get(OidcConsumedToken, "oidc-expired") is not None
        monkeypatch.setattr(oidc_google, "_utcnow", lambda: now + timedelta(days=1))
        assert oidc_google._consume_exchange_code(
            db, oidc_google._create_exchange_code(1)
        )
        assert db.get(OidcConsumedToken, "oauth-state:live") is not None
        assert not claim_oauth_state(db, "fresh", expires_at=time.time() + 600)


def test_previous_raw_token_claim_rejects_encoded_variant(sessions):
    state = auth.create_access_token(
        {"type": "oauth_state"}, expires_delta=timedelta(minutes=10)
    )
    with sessions() as db:
        db.add(
            OidcConsumedToken(
                token_id="oauth-state:" + hashlib.sha256(state.encode()).hexdigest(),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
            )
        )
        db.commit()
        assert not claim_oauth_state(db, state + "=", expires_at=time.time() + 600)
