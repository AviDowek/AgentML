"""Email service for sending invitations and notifications."""
import logging
from typing import Optional

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None,
) -> bool:
    """Send an email.

    Returns True if sent successfully, False otherwise.
    """
    if not settings.smtp_host:
        logger.warning(f"SMTP not configured. Would have sent email to {to_email}: {subject}")
        return True  # Return True in dev mode so flow continues

    try:
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        message = MIMEMultipart("alternative")
        message["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
        message["To"] = to_email
        message["Subject"] = subject

        # Add plain text and HTML versions
        if text_content:
            message.attach(MIMEText(text_content, "plain"))
        message.attach(MIMEText(html_content, "html"))

        await aiosmtplib.send(
            message,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user or None,
            password=settings.smtp_password or None,
            start_tls=True,
        )
        logger.info(f"Email sent successfully to {to_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False


async def send_project_invite_email(
    to_email: str,
    inviter_name: str,
    project_name: str,
    invite_token: str,
    role: str,
) -> bool:
    """Send a project sharing invitation email."""
    invite_url = f"{settings.frontend_url}/accept-invite?token={invite_token}&type=project"

    subject = f"{inviter_name} invited you to collaborate on '{project_name}'"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .button {{ display: inline-block; padding: 12px 24px; background-color: #6366f1;
                      color: white; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
            .button:hover {{ background-color: #5558e5; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>You've been invited to collaborate!</h2>
            <p><strong>{inviter_name}</strong> has invited you to collaborate on the project
               <strong>"{project_name}"</strong> as a <strong>{role}</strong>.</p>

            <p>Click the button below to accept the invitation and start collaborating:</p>

            <a href="{invite_url}" class="button">Accept Invitation</a>

            <p>If you don't have an account yet, you'll be able to create one when you accept
               the invitation.</p>

            <div class="footer">
                <p>If you didn't expect this invitation, you can safely ignore this email.</p>
                <p>This invitation was sent from AgentML.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    You've been invited to collaborate!

    {inviter_name} has invited you to collaborate on the project "{project_name}" as a {role}.

    Click the link below to accept the invitation:
    {invite_url}

    If you don't have an account yet, you'll be able to create one when you accept the invitation.

    If you didn't expect this invitation, you can safely ignore this email.
    """

    return await send_email(to_email, subject, html_content, text_content)
