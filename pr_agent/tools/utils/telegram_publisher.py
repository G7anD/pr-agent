"""Telegram Bot API sender — stdlib-only, no external HTTP deps.

Pattern adapted from `openclaw_jira_automation.telegram_deliver` (already in
production on the same Caretech server, using the same shared bot token).
"""

from __future__ import annotations

import json
import re
from typing import Optional
from urllib import request


class TelegramDeliveryError(RuntimeError):
    """Raised when the Telegram Bot API returns ok=false or the HTTP call fails."""


# Per Telegram MarkdownV2 spec, these chars MUST be backslash-escaped anywhere
# they appear outside of code/pre/link constructs.
# https://core.telegram.org/bots/api#markdownv2-style
_MD_V2_SPECIALS = r"_*[]()~`>#+-=|{}.!\\"
_MD_V2_RE = re.compile("([" + re.escape(_MD_V2_SPECIALS) + "])")


def escape_markdown_v2(text: str) -> str:
    """Escape all MarkdownV2 special chars with backslash.

    NOTE: This is for plain-text content. Do NOT apply to text that already
    contains intended Markdown formatting (e.g. `[text](url)` links the model
    produced) — those constructs have their own escape rules.
    """
    return _MD_V2_RE.sub(r"\\\1", text)


class TelegramPublisher:
    def __init__(self, bot_token: str, timeout: int = 10):
        self.bot_token = bot_token
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "MarkdownV2",
        disable_web_page_preview: bool = False,
        message_thread_id: Optional[int] = None,
    ) -> dict:
        payload = {
            "chat_id": str(chat_id),
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise TelegramDeliveryError(str(exc)) from exc
        if not data.get("ok"):
            raise TelegramDeliveryError(
                data.get("description") or "telegram sendMessage failed"
            )
        return data.get("result", {})

    def send_photo(
        self,
        chat_id: str,
        photo_bytes: bytes,
        caption: Optional[str] = None,
        parse_mode: str = "MarkdownV2",
        disable_notification: bool = False,
        filename: str = "banner.png",
    ) -> dict:
        """Send a photo via multipart/form-data sendPhoto.

        Hand-rolled multipart (stdlib only) — no `requests` dependency. The
        boundary is a fixed unusual token; PNG bytes will not contain it.
        """
        boundary = "----AuroraBannerBoundary7MA4YWxkTrZu0gW"
        crlf = b"\r\n"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(f"--{boundary}".encode() + crlf)
            parts.append(f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf)
            parts.append(value.encode("utf-8") + crlf)

        add_field("chat_id", str(chat_id))
        if caption is not None:
            add_field("caption", caption)
            add_field("parse_mode", parse_mode)
        if disable_notification:
            add_field("disable_notification", "true")

        # file part
        parts.append(f"--{boundary}".encode() + crlf)
        parts.append(
            f'Content-Disposition: form-data; name="photo"; filename="{filename}"'.encode() + crlf
        )
        parts.append(b"Content-Type: image/png" + crlf + crlf)
        parts.append(photo_bytes + crlf)
        parts.append(f"--{boundary}--".encode() + crlf)

        body = b"".join(parts)
        req = request.Request(
            f"{self.base_url}/sendPhoto",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise TelegramDeliveryError(str(exc)) from exc
        if not data.get("ok"):
            raise TelegramDeliveryError(data.get("description") or "telegram sendPhoto failed")
        return data.get("result", {})
