"""Durable single-use claims for signed, browser-bound ordinary OAuth states."""

import hashlib
import math
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ..models.oidc_consumed_token import OidcConsumedToken


def claim_oauth_state(db: Session, state: str, *, expires_at: object) -> bool:
    """Commit consumption before exchanging a code, across workers/restarts.

    The caller must validate the signed state and its browser cookie first.
    Reuse the existing short-lived consumed-token table with a disjoint key
    namespace. A rollback of later provider work must never undo this claim.
    This is a transaction boundary: call before making account changes.
    """
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return False
    try:
        if not math.isfinite(expires_at):
            return False
        expiry = datetime.fromtimestamp(expires_at, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return False
    token_id = "oauth-state:" + hashlib.sha256(state.encode()).hexdigest()
    try:
        # A shared clock must govern both retention and admission. A slow API
        # replica must not reclaim a proof another replica already expired.
        now = db.scalar(select(func.current_timestamp()))
        if now is None:
            raise RuntimeError("OAuth state store did not return its clock")
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if expiry <= now:
            return False
        db.query(OidcConsumedToken).filter(OidcConsumedToken.expires_at <= now).delete(
            synchronize_session=False
        )
        db.add(OidcConsumedToken(token_id=token_id, expires_at=expiry))
        db.commit()
    except IntegrityError:
        db.rollback()
        return False
    except SQLAlchemyError:
        db.rollback()
        raise
    return True
