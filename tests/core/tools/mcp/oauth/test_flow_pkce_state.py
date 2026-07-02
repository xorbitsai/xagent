import pytest

from xagent.core.tools.core.mcp.oauth.flow import (
    decode_state,
    encode_state,
    new_pkce,
)


def test_pkce_pair():
    verifier, challenge = new_pkce()
    assert 43 <= len(verifier) <= 128
    assert challenge and challenge != verifier


def test_state_roundtrip():
    token = encode_state(user_id=3, mcpserver_id=9)
    user_id, server_id, nonce = decode_state(token)
    assert (user_id, server_id) == (3, 9)
    assert nonce


def test_state_tamper_detected():
    token = encode_state(user_id=3, mcpserver_id=9)
    with pytest.raises(ValueError):
        decode_state(token + "tamper")


def test_state_malformed_payload_raises_value_error():
    # A payload that decrypts successfully but has the wrong JSON shape must
    # still raise ValueError (not KeyError), so callers can catch one type.
    from xagent.core.utils.encryption import get_cipher

    bad = get_cipher().encrypt(b'{"u": 1}').decode()  # missing "s" and "n"
    with pytest.raises(ValueError):
        decode_state(bad)
