from unittest.mock import MagicMock

import pytest

from xagent.web.services import auth_email


def test_confirmation_email_uses_configured_expiry(monkeypatch):
    monkeypatch.setenv("XAGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("XAGENT_SMTP_FROM_EMAIL", "sender@example.com")
    monkeypatch.setenv("XAGENT_EMAIL_CONFIRMATION_EXPIRE_MINUTES", "7")
    smtp = MagicMock()
    monkeypatch.setattr(auth_email.smtplib, "SMTP", smtp)
    auth_email.send_email_change_confirmation(
        "target@example.com", "opaque-code", "Test"
    )
    message = smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
    assert message["To"] == "target@example.com"
    assert "7 minutes" in message.get_content()
    assert "opaque-code" in message.get_content()


@pytest.mark.parametrize("missing", ["HOST", "FROM_EMAIL"])
def test_account_mail_requires_sender_and_host(monkeypatch, missing):
    monkeypatch.setenv("XAGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("XAGENT_SMTP_FROM_EMAIL", "sender@example.com")
    monkeypatch.delenv(f"XAGENT_SMTP_{missing}")
    assert not auth_email.is_auth_email_configured()
    with pytest.raises(RuntimeError, match="SMTP is not configured for account emails"):
        auth_email.send_email_change_confirmation("target@example.com", "code", "Test")
