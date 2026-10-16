import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

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
    db.scalar.side_effect = OperationalError("statement", {}, Exception("unavailable"))
    with pytest.raises(OperationalError):
        claim_oauth_state(db, "state", expires_at=time.time() + 600)
    db.rollback.assert_called_once()
