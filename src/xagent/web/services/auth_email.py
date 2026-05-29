import smtplib
from email.message import EmailMessage

from ...config import (
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
    smtp_host = get_smtp_host()
    smtp_port = get_smtp_port()
    smtp_username = get_smtp_username()
    smtp_password = get_smtp_password()
    smtp_use_tls = get_smtp_use_tls()
    smtp_use_ssl = get_smtp_use_ssl()
    from_email = get_password_reset_email_sender()
    from_name = get_smtp_from_name(app_name)

    if not smtp_host or not from_email:
        raise RuntimeError("SMTP is not configured for password reset emails")

    message = EmailMessage()
    message["Subject"] = get_password_reset_email_subject(app_name)
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                f"You requested a password reset for {app_name}.",
                "",
                "Open the link below to set a new password:",
                reset_link,
                "",
                (
                    "This link expires in "
                    f"{get_password_reset_expire_minutes()} minutes."
                ),
                "If you did not request this, you can ignore this email.",
            ]
        )
    )

    smtp_client_cls = smtplib.SMTP_SSL if smtp_use_ssl else smtplib.SMTP
    with smtp_client_cls(smtp_host, smtp_port, timeout=10) as server:
        server.ehlo()
        if smtp_use_tls and not smtp_use_ssl:
            server.starttls()
            server.ehlo()
        if smtp_username:
            server.login(smtp_username, smtp_password)
        server.send_message(message)
