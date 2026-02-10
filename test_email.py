import os

from services.mailer import send_email_smtp


def main():
    to_email = os.environ.get("TEST_TO_EMAIL", "").strip()
    if not to_email:
        raise RuntimeError("Please set TEST_TO_EMAIL env var")
    send_email_smtp(
        to_email=to_email,
        subject="FX Flask 邮件测试",
        text_body="这是一封测试邮件：SMTP 配置正常。",
        html_body="<p>这是一封<b>测试邮件</b>：SMTP 配置正常。</p>",
    )
    print("sent")


if __name__ == "__main__":
    main()
