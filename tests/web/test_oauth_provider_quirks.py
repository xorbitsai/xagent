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


def test_meta_invalid_token_error_code_normalizes_code_190():
    error = {
        "message": "Error validating access token: Session has expired.",
        "type": "OAuthException",
        "code": 190,
    }
    assert meta_invalid_token_error_code(error) == "invalid_grant"


def test_meta_invalid_token_error_code_ignores_other_oauth_exceptions():
    """A different OAuthException code (e.g. a permission error) is not the
    "token is dead" signal -- only code 190 is."""
    error = {"message": "Missing permission", "type": "OAuthException", "code": 10}
    assert meta_invalid_token_error_code(error) is None


def test_meta_invalid_token_error_code_rejects_non_mapping_and_non_oauth_shapes():
    assert meta_invalid_token_error_code("invalid_grant") is None
    assert meta_invalid_token_error_code(None) is None
    assert (
        meta_invalid_token_error_code({"type": "OtherException", "code": 190}) is None
    )
