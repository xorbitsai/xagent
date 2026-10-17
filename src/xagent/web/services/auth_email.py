import smtplib
from email.message import EmailMessage

from ...config import (
    get_email_confirmation_expire_minutes,
    get_password_reset_expire_minutes,
    get_smtp_from_email,
    get_smtp_from_name,
    get_smtp_host,
    get_smtp_password,
    get_smtp_port,
    get_smtp_use_ssl,
    get_smtp_use_tls,
    get_smtp_username,
)


def get_password_reset_email_sender() -> str:
    return get_smtp_from_email()


def get_password_reset_email_subject(app_name: str) -> str:
    return f"Reset your {app_name} password"


def send_password_reset_email(to_email: str, reset_link: str, app_name: str) -> None:
    message = EmailMessage()
    message["Subject"] = get_password_reset_email_subject(app_name)
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                f"You requested a password reset for {app_name}.",
                "",
                "Open the link below to set a new password:",
                reset_link,
                "",
                f"This link expires in {get_password_reset_expire_minutes()} minutes.",
                "If you did not request this, you can ignore this email.",
            ]
        )
    )
    _send_auth_message(message, app_name)


def send_email_change_confirmation(to_email: str, token: str, app_name: str) -> None:
    message = EmailMessage()
    message["Subject"] = f"Confirm your {app_name} email address"
    message["To"] = to_email
    message.set_content(
        f"Enter this confirmation code in {app_name} Settings:\n\n{token}\n\n"
        f"This code expires in {get_email_confirmation_expire_minutes()} minutes. "
        "Your email address will not change "
        "until you confirm. If you did not request this, ignore this email."
    )
    _send_auth_message(message, app_name)


def is_auth_email_configured() -> bool:
    """Configuration readiness only; actual SMTP delivery may still fail."""
    return bool(get_smtp_host() and get_smtp_from_email())


def send_email_changed_notice(to_email: str, app_name: str) -> None:
    message = EmailMessage()
    message["Subject"] = f"Your {app_name} recovery email changed"
    message["To"] = to_email
    message.set_content(
        f"Your {app_name} account recovery email address has been changed.\n\n"
        "If you did not request this, contact your administrator and reset "
        "your password immediately."
    )
    _send_auth_message(message, app_name)


def _send_auth_message(message: EmailMessage, app_name: str) -> None:
    smtp_host = get_smtp_host()
    smtp_port = get_smtp_port()
    smtp_username = get_smtp_username()
    smtp_password = get_smtp_password()
    smtp_use_tls = get_smtp_use_tls()
    smtp_use_ssl = get_smtp_use_ssl()
    from_email = get_password_reset_email_sender()
    from_name = get_smtp_from_name(app_name)

    if not smtp_host or not from_email:
        raise RuntimeError("SMTP is not configured for account emails")

    message["From"] = f"{from_name} <{from_email}>"

    smtp_client_cls = smtplib.SMTP_SSL if smtp_use_ssl else smtplib.SMTP
    with smtp_client_cls(smtp_host, smtp_port, timeout=10) as server:
        server.ehlo()
        if smtp_use_tls and not smtp_use_ssl:
            server.starttls()
            server.ehlo()
        if smtp_username:
            server.login(smtp_username, smtp_password)
        server.send_message(message)
