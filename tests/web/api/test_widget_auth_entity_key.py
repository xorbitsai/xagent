"""Unit tests for the widget auth rate-limit entity-key derivation (#1108).

``_widget_auth_rate_limit_entity_key`` picks the loose-entity backstop bucket
for the widget auth / embed-ticket gate from a *signed* ticket's claims (no DB
work), so a per-page-load ticket rotation cannot escape it. These pin each
branch — agent ticket, workforce ticket, legacy (no owner_type), tampered /
wrong-type / malformed tickets, the direct widget-key flow, and the empty
fallback — directly, without standing up the HTTP widget flow.
"""

from __future__ import annotations

from datetime import timedelta

from xagent.web.api.auth import create_access_token
from xagent.web.api.widget import (
    EMBED_TICKET_OWNER_WORKFORCE,
    EMBED_TICKET_TYPE,
    _widget_auth_rate_limit_entity_key,
)


def _mint(claims: dict) -> str:
    return create_access_token(claims, expires_delta=timedelta(seconds=60))


def test_agent_ticket_keys_on_agent_entity() -> None:
    # No owner_type is the agent branch, which is also the legacy shape: tickets
    # minted before workforce support carried only agent_id and no owner_type,
    # so this covers both (missing and agent owner_type take the same branch).
    ticket = _mint({"type": EMBED_TICKET_TYPE, "agent_id": 42})
    assert _widget_auth_rate_limit_entity_key(ticket, None) == "agent:42"


def test_workforce_ticket_keys_on_workforce_entity() -> None:
    ticket = _mint(
        {
            "type": EMBED_TICKET_TYPE,
            "owner_type": EMBED_TICKET_OWNER_WORKFORCE,
            "workforce_id": 7,
        }
    )
    assert _widget_auth_rate_limit_entity_key(ticket, None) == "workforce:7"


def test_tampered_ticket_collapses_to_invalid_bucket() -> None:
    ticket = _mint({"type": EMBED_TICKET_TYPE, "agent_id": 42})
    assert (
        _widget_auth_rate_limit_entity_key(ticket + "tamper", None) == "invalid-ticket"
    )


def test_wrong_type_ticket_collapses_to_invalid_bucket() -> None:
    ticket = _mint({"type": "not-an-embed-ticket", "agent_id": 42})
    assert _widget_auth_rate_limit_entity_key(ticket, None) == "invalid-ticket"


def test_workforce_ticket_without_id_collapses_to_invalid_bucket() -> None:
    ticket = _mint(
        {"type": EMBED_TICKET_TYPE, "owner_type": EMBED_TICKET_OWNER_WORKFORCE}
    )
    assert _widget_auth_rate_limit_entity_key(ticket, None) == "invalid-ticket"


def test_direct_widget_key_flow_keys_on_prefixed_key() -> None:
    assert _widget_auth_rate_limit_entity_key(None, "wk_abc") == "key:wk_abc"


def test_ticket_wins_over_widget_key() -> None:
    ticket = _mint({"type": EMBED_TICKET_TYPE, "agent_id": 9})
    assert _widget_auth_rate_limit_entity_key(ticket, "wk_abc") == "agent:9"


def test_no_credential_returns_empty_shared_fallback() -> None:
    assert _widget_auth_rate_limit_entity_key(None, None) == ""
