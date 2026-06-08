from unittest.mock import patch, MagicMock

from pr_agent.tools.utils.banner_client import fetch_banner


class TestFetchBanner:
    def test_returns_png_bytes_on_success(self):
        with patch("pr_agent.tools.utils.banner_client.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b"\x89PNG\r\n\x1a\nFAKEDATA"
            urlopen.return_value.__enter__.return_value = mock_resp

            data = fetch_banner("http://172.17.0.1:41928", old="2026.06.10", new="26.6.1")

            assert data == b"\x89PNG\r\n\x1a\nFAKEDATA"
            called_url = urlopen.call_args[0][0]
            assert called_url.startswith("http://172.17.0.1:41928/banner?")
            assert "new=26.6.1" in called_url
            assert "old=2026.06.10" in called_url
            assert "lang=ru" in called_url

    def test_returns_none_on_empty_body(self):
        with patch("pr_agent.tools.utils.banner_client.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b""
            urlopen.return_value.__enter__.return_value = mock_resp

            assert fetch_banner("http://x", old="a", new="b") is None

    def test_returns_none_on_http_error(self):
        from urllib.error import HTTPError
        with patch("pr_agent.tools.utils.banner_client.request.urlopen") as urlopen:
            urlopen.side_effect = HTTPError(url="x", code=500, msg="err", hdrs=None, fp=None)
            assert fetch_banner("http://x", old="a", new="b") is None

    def test_returns_none_on_network_error(self):
        with patch("pr_agent.tools.utils.banner_client.request.urlopen") as urlopen:
            urlopen.side_effect = OSError("connection refused")
            assert fetch_banner("http://x", old="a", new="b") is None

    def test_handles_trailing_slash_in_service_url(self):
        with patch("pr_agent.tools.utils.banner_client.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b"PNGDATA"
            urlopen.return_value.__enter__.return_value = mock_resp
            fetch_banner("http://x:41928/", old="a", new="b")
            called_url = urlopen.call_args[0][0]
            assert called_url.startswith("http://x:41928/banner?")
