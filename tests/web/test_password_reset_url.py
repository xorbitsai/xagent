from urllib.parse import parse_qs, urlparse

from xagent.web.api.auth import build_password_reset_url


def test_password_reset_token_is_only_in_fragment(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://app.example.com")

    parsed = urlparse(build_password_reset_url("token-value"))

    assert parsed.query == ""
    assert parse_qs(parsed.fragment) == {"token": ["token-value"]}
