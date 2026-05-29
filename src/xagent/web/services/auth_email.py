import os
import smtplib
from email.message import EmailMessage


def get_password_reset_email_sender() -> str:
    return os.getenv("XAGENT_SMTP_FROM_EMAIL", "").strip()


def get_password_reset_email_subject(app_name: str) -> str:
    return f"Reset your {app_name} password"


def send_password_reset_email(to_email: str, reset_link: str, app_name: str) -> None:
    smtp_host = os.getenv("XAGENT_SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("XAGENT_SMTP_PORT", "587"))
    smtp_username = os.getenv("XAGENT_SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("XAGENT_SMTP_PASSWORD", "")
    smtp_use_tls = os.getenv("XAGENT_SMTP_USE_TLS", "true").lower() == "true"
    smtp_use_ssl = os.getenv("XAGENT_SMTP_USE_SSL", "false").lower() == "true"
    from_email = get_password_reset_email_sender()
    from_name = os.getenv("XAGENT_SMTP_FROM_NAME", app_name).strip() or app_name

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
                "This link expires in 30 minutes.",
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
