from xagent.web.oauth_provider_quirks import requires_json_accept_header


def test_requires_json_accept_header_true_for_github():
    assert requires_json_accept_header("github") is True


def test_requires_json_accept_header_is_case_insensitive():
    assert requires_json_accept_header("GitHub") is True
    assert requires_json_accept_header("GITHUB") is True


def test_requires_json_accept_header_false_for_other_providers():
    for provider in ("zoom", "slack", "google", "linkedin", "hubspot", "intercom"):
        assert requires_json_accept_header(provider) is False
