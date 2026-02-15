from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Optional


def send_email_smtp(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
) -> None:
    """Send an email using SMTP settings from environment variables.

    Required env:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS

    Optional env:
      SMTP_FROM (default SMTP_USER)
      SMTP_USE_TLS (default true)
      SMTP_USE_SSL (default false)
    """
    # host = os.environ.get("SMTP_HOST", "").strip()
    # port = int(os.environ.get("SMTP_PORT", "587"))
    # user = os.environ.get("SMTP_USER", "").strip()
    # pwd = os.environ.get("SMTP_PASS", "").strip()

    host = "smtp.qq.com"
    pwd = "tqlkjkqxysuybfbj"
    user = "dododeyx@qq.com"
    port = "465"

    if not host or not user or not pwd:
        raise RuntimeError("SMTP 未配置：请设置 SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS 环境变量")

    from_email = os.environ.get("SMTP_FROM", user).strip()
    use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() in ("1", "true", "yes", "y", "on")
    use_ssl = os.environ.get("SMTP_USE_SSL", "false").strip().lower() in ("1", "true", "yes", "y", "on")

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    if use_ssl:
        with smtplib.SMTP_SSL(host, port) as s:
            s.login(user, pwd)
            s.send_message(msg)
        return

    with smtplib.SMTP(host, port) as s:
        s.ehlo()
        if use_tls:
            s.starttls()
            s.ehlo()
        s.login(user, pwd)
        s.send_message(msg)
