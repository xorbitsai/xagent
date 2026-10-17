"""Durable single-use navigation proofs, separate from access and OAuth state."""

import math
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ..models.oidc_consumed_token import OidcConsumedToken

_PREFIX = "oauth-launch:"


def consume_oauth_launch_ticket(db: Session, claims: dict[str, Any]) -> bool:
    """Caller verifies signature, purpose, provider and current user first.

    Keep the claim committed if later launch setup fails, including across API
    workers and restarts. Both admission and cleanup use the database clock.
    """
    token_id = claims.get("jti")
    expiry = claims.get("exp")
    if not isinstance(token_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{32}", token_id
    ):
        return False
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)):
        return False
    try:
        if not math.isfinite(expiry):
            return False
        expires_at = datetime.fromtimestamp(expiry, timezone.utc)
    except (ValueError, OverflowError, OSError):
        return False
    try:
        now = db.execute(select(func.current_timestamp())).scalar_one()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            return False
        db.query(OidcConsumedToken).filter(
            OidcConsumedToken.token_id.startswith(_PREFIX),
            OidcConsumedToken.expires_at <= now,
        ).delete(synchronize_session=False)
        db.add(OidcConsumedToken(token_id=_PREFIX + token_id, expires_at=expires_at))
        db.commit()
    except IntegrityError:
        db.rollback()
        return False
    except SQLAlchemyError:
        db.rollback()
        raise
    return True
