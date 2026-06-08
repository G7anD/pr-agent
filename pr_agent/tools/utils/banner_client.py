"""HTTP client for the banner-service.

The banner-service (Node + Puppeteer) renders the Aurora+ release banner PNG.
This client fetches it. All failures return None — banner is best-effort and
the release notes must publish regardless.
"""

from __future__ import annotations

from typing import Optional
from urllib import request, parse

from pr_agent.log import get_logger


def fetch_banner(
    service_url: str,
    old: str,
    new: str,
    lang: str = "ru",
    timeout: int = 30,
) -> Optional[bytes]:
    """GET {service_url}/banner?old=&new=&lang= → PNG bytes, or None on any failure."""
    qs = parse.urlencode({"old": old or "", "new": new, "lang": lang})
    url = f"{service_url.rstrip('/')}/banner?{qs}"
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
            if getattr(resp, "status", 200) == 200 and data:
                return data
            get_logger().warning(f"banner_client: empty/non-200 from {url}")
            return None
    except Exception as e:
        get_logger().warning(f"banner_client: fetch failed for {url} — {e}")
        return None
