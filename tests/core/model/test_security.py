"""Tests for security redaction helpers used by model integrations."""

from xagent.core.utils.security import (
    redact_sensitive_text,
    redact_url_credentials_for_logging,
)


def test_redact_url_credentials_for_logging_masks_sensitive_query_values() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecret&v=1"
    redacted = redact_url_credentials_for_logging(url)

    assert "AIzaSySecret" not in redacted
    assert "key=%2A%2A%2A" in redacted
    assert "v=1" in redacted


def test_redact_sensitive_text_masks_bearer_and_header_keys() -> None:
    text = (
        "Authorization: Bearer sk-secret-value "
        "x-goog-api-key: AIzaSyHeaderSecret "
        "url=https://example.com/path?api_key=my_api_key"
    )
    redacted = redact_sensitive_text(text)

    assert "sk-secret-value" not in redacted
    assert "AIzaSyHeaderSecret" not in redacted
    assert "my_api_key" not in redacted


def test_redact_sensitive_text_masks_assignment_style_secrets() -> None:
    text = "api_key=sk-super-secret timeout=30"
    redacted = redact_sensitive_text(text)

    assert "sk-super-secret" not in redacted
    assert "api_key=***" in redacted
    assert "timeout=30" in redacted
