import json
from unittest.mock import patch, MagicMock

import pytest

from pr_agent.tools.utils.telegram_publisher import (
    TelegramPublisher,
    TelegramDeliveryError,
    escape_markdown_v2,
)


class TestEscapeMarkdownV2:
    @pytest.mark.parametrize("raw,expected", [
        ("hello", "hello"),
        ("hello.world", r"hello\.world"),
        ("v2026.06.7", r"v2026\.06\.7"),
        ("a-b-c", r"a\-b\-c"),
        ("(x)", r"\(x\)"),
        ("[x]", r"\[x\]"),
        ("a + b = c!", r"a \+ b \= c\!"),
        ("under_score", r"under\_score"),
        ("*bold*", r"\*bold\*"),
        ("a > b", r"a \> b"),
        ("a # b", r"a \# b"),
        ("{json}", r"\{json\}"),
        ("a|b", r"a\|b"),
        ("a`b", r"a\`b"),
        ("a~b", r"a\~b"),
        ("a\\b", r"a\\b"),
    ])
    def test_escapes_all_specials(self, raw, expected):
        assert escape_markdown_v2(raw) == expected

    def test_does_not_escape_emoji(self):
        assert escape_markdown_v2("🚀💉🦷") == "🚀💉🦷"


class TestTelegramPublisher:
    def test_send_message_posts_to_telegram_api(self):
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"ok": True, "result": {"message_id": 42}}
            ).encode()
            urlopen.return_value.__enter__.return_value = mock_resp

            pub = TelegramPublisher(bot_token="123:abc")
            result = pub.send_message(chat_id="-100xyz", text="hi", parse_mode="MarkdownV2")

            assert result["message_id"] == 42
            # Verify request URL and payload
            req = urlopen.call_args[0][0]
            assert req.full_url == "https://api.telegram.org/bot123:abc/sendMessage"
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["chat_id"] == "-100xyz"
            assert payload["text"] == "hi"
            assert payload["parse_mode"] == "MarkdownV2"

    def test_raises_on_api_failure(self):
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"ok": False, "description": "Bad Request: chat not found"}
            ).encode()
            urlopen.return_value.__enter__.return_value = mock_resp

            pub = TelegramPublisher(bot_token="123:abc")
            with pytest.raises(TelegramDeliveryError, match="chat not found"):
                pub.send_message(chat_id="-100xyz", text="hi")

    def test_raises_on_network_failure(self):
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            urlopen.side_effect = OSError("connection refused")

            pub = TelegramPublisher(bot_token="123:abc")
            with pytest.raises(TelegramDeliveryError, match="connection refused"):
                pub.send_message(chat_id="-100xyz", text="hi")


class TestSendPhoto:
    def test_send_photo_posts_multipart(self):
        from pr_agent.tools.utils.telegram_publisher import TelegramPublisher
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"ok": True, "result": {"message_id": 7}}
            ).encode()
            urlopen.return_value.__enter__.return_value = mock_resp

            pub = TelegramPublisher(bot_token="123:abc")
            result = pub.send_photo(
                chat_id="-100xyz",
                photo_bytes=b"\x89PNGDATA",
                caption="hello",
                parse_mode="MarkdownV2",
            )

            assert result["message_id"] == 7
            req = urlopen.call_args[0][0]
            assert req.full_url == "https://api.telegram.org/bot123:abc/sendPhoto"
            ctype = req.headers["Content-type"]
            assert ctype.startswith("multipart/form-data; boundary=")
            body = req.data
            assert b'name="chat_id"' in body
            assert b"-100xyz" in body
            assert b'name="caption"' in body
            assert b"hello" in body
            assert b'name="parse_mode"' in body
            assert b"MarkdownV2" in body
            assert b'name="photo"; filename=' in body
            assert b"\x89PNGDATA" in body

    def test_send_photo_without_caption_omits_caption_field(self):
        from pr_agent.tools.utils.telegram_publisher import TelegramPublisher
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"ok": True, "result": {"message_id": 9}}).encode()
            urlopen.return_value.__enter__.return_value = mock_resp
            pub = TelegramPublisher(bot_token="123:abc")
            pub.send_photo(chat_id="-100xyz", photo_bytes=b"PNG")
            body = urlopen.call_args[0][0].data
            assert b'name="caption"' not in body

    def test_send_photo_raises_on_api_failure(self):
        from pr_agent.tools.utils.telegram_publisher import TelegramPublisher, TelegramDeliveryError
        with patch("pr_agent.tools.utils.telegram_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"ok": False, "description": "PHOTO_INVALID_DIMENSIONS"}
            ).encode()
            urlopen.return_value.__enter__.return_value = mock_resp
            pub = TelegramPublisher(bot_token="123:abc")
            with pytest.raises(TelegramDeliveryError, match="PHOTO_INVALID_DIMENSIONS"):
                pub.send_photo(chat_id="-100xyz", photo_bytes=b"PNG")
