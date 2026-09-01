import pytest

from xagent.web.oauth_provider_quirks import (
    meta_invalid_token_error_code,
    requires_json_accept_header,
)


def test_requires_json_accept_header_true_for_github():
    assert requires_json_accept_header("github") is True


def test_requires_json_accept_header_is_case_insensitive():
    assert requires_json_accept_header("GitHub") is True
    assert requires_json_accept_header("GITHUB") is True


def test_requires_json_accept_header_false_for_other_providers():
    for provider in ("zoom", "slack", "google", "linkedin", "hubspot", "intercom"):
        assert requires_json_accept_header(provider) is False


@pytest.mark.parametrize("code", [190, 102])
def test_meta_invalid_token_error_code_normalizes_dead_session_codes(code):
    """190 (invalid/expired access token) and 102 (session key invalid or
    no longer valid) are Meta's two distinct top-level OAuthException codes
    for a dead session, not merely a transient blip."""
    error = {
        "message": "Error validating access token: Session has expired.",
        "type": "OAuthException",
        "code": code,
    }
    assert meta_invalid_token_error_code(error) == "invalid_grant"


def test_meta_invalid_token_error_code_normalizes_code_190_regardless_of_subcode():
    """Meta nests session-invalidation detail (password changed, expired,
    logged out) as `error_subcode` under the top-level code 190 -- e.g.
    {"code": 190, "error_subcode": 463} -- never as a bare top-level code
    of 463/467 itself. code 190 alone must already cover every subcode
    without needing to inspect error_subcode."""
    error = {
        "message": "Error validating access token: Session has expired.",
        "type": "OAuthException",
        "code": 190,
        "error_subcode": 463,
    }
    assert meta_invalid_token_error_code(error) == "invalid_grant"


def test_meta_invalid_token_error_code_ignores_other_oauth_exceptions():
    """A different OAuthException code (e.g. a permission error) is not one
    of the "session is dead" signals in _META_INVALID_TOKEN_ERROR_CODES."""
    error = {"message": "Missing permission", "type": "OAuthException", "code": 10}
    assert meta_invalid_token_error_code(error) is None


def test_meta_invalid_token_error_code_rejects_non_mapping_and_non_oauth_shapes():
    assert meta_invalid_token_error_code("invalid_grant") is None
    assert meta_invalid_token_error_code(None) is None
    assert (
        meta_invalid_token_error_code({"type": "OtherException", "code": 190}) is None
    )
