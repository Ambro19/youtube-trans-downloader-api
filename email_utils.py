# backend/email_utils.py
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")  # set in Render → Environment
SENDGRID_FROM    = os.getenv("CONTACT_FROM", "no-reply@onetechly.com")  # verified sender
SENDGRID_NAME    = os.getenv("CONTACT_FROM_NAME", "OneTechly")          # optional display name

def send_password_reset_email(to_email: str, reset_link: str) -> Optional[str]:
    """
    Sends a password reset email via SendGrid if SENDGRID_API_KEY is set.
    Returns the SendGrid message ID on success, or None if we fell back to logging.
    Raises on hard SendGrid errors.
    """
    if not SENDGRID_API_KEY:
        log.info("[PWD-RESET] (no SENDGRID_API_KEY) Send this link to the user: %s", reset_link)
        return None

    # Lazy import so the app still runs without sendgrid installed locally
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, From, To
    except Exception as e:
        log.warning("sendgrid package not installed; logging reset link. %s", e)
        log.info("[PWD-RESET] %s", reset_link)
        return None

    subject = "Reset your OneTechly password"
    html = f"""
    <p>Hello,</p>
    <p>We received a request to reset your password. Click the button below:</p>
    <p><a href="{reset_link}"
          style="display:inline-block;padding:10px 16px;background:#4f46e5;color:#fff;
                 border-radius:8px;text-decoration:none">Reset password</a></p>
    <p>If you didn’t request this, you can ignore this email.</p>
    <p>— The OneTechly team</p>
    """

    msg = Mail(
        from_email=From(SENDGRID_FROM, SENDGRID_NAME),
        to_emails=To(to_email),
        subject=subject,
        html_content=html,
    )

    sg = SendGridAPIClient(SENDGRID_API_KEY)
    resp = sg.send(msg)

    if 200 <= resp.status_code < 300:
        msg_id = resp.headers.get("X-Message-Id") or resp.headers.get("X-Message-ID")
        log.info("Password reset email queued to %s (SendGrid %s)", to_email, msg_id or "OK")
        return msg_id

    # Non-2xx: raise so the caller can decide what to do
    raise RuntimeError(f"SendGrid error {resp.status_code}: {resp.body}")
