"""Malformed signed dates remain credential rejections at JWT consumers."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.web import jwt_codec as jwt
from xagent.web.api import auth, widget
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY


@pytest.mark.parametrize("verify", [auth.verify_refresh_token, auth.verify_token])
@pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
@pytest.mark.parametrize("value", [None, [], {}, float("inf"), "not-a-number"])
def test_auth_helpers_reject_malformed_temporal_claims(verify, claim, value):
    token = jwt.encode(
        {"type": "refresh", "user_id": 1, "exp": 4102444800, claim: value},
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )
    assert verify(token) is None


@pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
@pytest.mark.parametrize("value", [None, [], {}, float("inf"), "not-a-number"])
def test_embed_ticket_rejects_malformed_dates_before_database_query(claim, value):
    token = jwt.encode(
        {
            "type": widget.EMBED_TICKET_TYPE,
            "agent_id": 1,
            "embed_origin": "https://example.com",
            "exp": 4102444800,
            claim: value,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )
    db = Mock(spec=Session)
    with pytest.raises(HTTPException) as raised:
        widget._owner_from_embed_ticket(db, token)
    assert raised.value.status_code == 403
    assert raised.value.detail == "Invalid or expired embed ticket"
    db.query.assert_not_called()
