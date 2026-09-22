"""
email_parser.py
Parses raw pasted email text (headers + body) into structured fields:
sender, sender_display_name, sender_domain, subject, body, urls.
receiver/date are deliberately excluded — see notebook notes on leakage risk.
"""

import re
from email import message_from_string
from email.utils import parseaddr


def parse_raw_email(raw_text: str) -> dict:
    """
    Takes the full raw email text (as pasted by the user, including headers)
    and returns a dict with sender, sender_display_name, sender_domain,
    subject, body, urls.
    """
    msg = message_from_string(raw_text)

    sender = msg.get("From", "")
    subject = msg.get("Subject", "")
    # receiver and date are intentionally NOT extracted for modeling —
    # they carry dataset-specific leakage risk (see notebook notes).

    display_name, sender_email = parseaddr(sender)
    sender_domain = sender_email.split("@")[-1].lower() if "@" in sender_email else ""
    # username (the part before '@') is deliberately discarded — not used as a feature

    body = _get_body(msg)
    urls = _extract_urls(body)

    return {
        "sender": sender.strip(),
        "sender_display_name": display_name.strip(),
        "sender_domain": sender_domain,
        "subject": subject.strip(),
        "body": body.strip(),
        "urls": urls,
    }


def _get_body(msg) -> str:
    """Extracts plain-text body from an email.Message object, handling multipart emails."""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode(errors="ignore"))
                except Exception:
                    continue
        return "\n".join(parts)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(errors="ignore")
        return msg.get_payload() or ""


def _extract_urls(text: str) -> list:
    """Extracts all URLs found in the body text."""
    url_pattern = re.compile(r'https?://[^\s"\'<>]+')
    return url_pattern.findall(text)


# Fallback for cases where the user pastes something that ISN'T
# a well-formed raw email (e.g. no headers, just subject/body copy-pasted).
# You may want to expand this depending on how "raw" your users' input actually is.
def parse_loose_text(raw_text: str) -> dict:
    """
    Fallback parser for when input doesn't have proper email headers.
    Treats the whole thing as body text, tries to guess subject as first line.
    """
    lines = raw_text.strip().split("\n")
    subject = lines[0] if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else raw_text
    urls = _extract_urls(raw_text)

    return {
        "sender": "",
        "sender_display_name": "",
        "sender_domain": "",
        "subject": subject,
        "body": body,
        "urls": urls,
    }